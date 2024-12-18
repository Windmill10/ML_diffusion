from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import librosa
import plotly.graph_objects as go
import plotly.graph_objs as go
import pytorch_lightning as pl
import torch
import torchaudio
import wandb
from audio_data_pytorch.utils import fractional_random_split
from audio_diffusion_pytorch import AudioDiffusionAutoencoder, Sampler, Schedule
from einops import rearrange
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from torch import LongTensor, Tensor, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_eps: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_weight_decay: float,
        num_tokens: int,
        num_features: int,
        max_length: int,
        tokenizer: str,
        autoencoder_path: str,  # Must be quantized
        transformer_decoder: nn.Module,
        transformer_encoder: nn.Module,
    ):
        super().__init__()
        self.lr = lr
        self.lr_eps = lr_eps
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_weight_decay = lr_weight_decay
        self.max_length = max_length

        self.transformer_decoder = transformer_decoder
        self.transformer_encoder = transformer_encoder
        self.autoencoder = torch.load(autoencoder_path, map_location=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer
        )
        self.to_embedding = nn.Embedding(num_tokens, num_features)

    def encode_text(self, texts: List[str]) -> Tensor:
        # Compute batch of tokens and mask from texts
        encoded = self.tokenizer.batch_encode_plus(
            [text.lower() for text in texts],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        tokens = encoded["input_ids"].to(self.device)
        # mask = encoded["attention_mask"].to(self.device).bool()
        # Compute embedding
        embedding = self.to_embedding(tokens)
        # Encode with transformer
        embedding_encoded = self.transformer_encoder(embedding)
        return embedding_encoded

    @torch.no_grad()
    def encode_audio(self, x: Tensor) -> Tensor:
        context, info = self.autoencoder.encode(x, with_info=True)
        return rearrange(info["indices"], "b 1 n r -> b n r")

    @torch.no_grad()
    def decode_audio(self, indices: Tensor) -> Tensor:
        indices = rearrange(indices, "b n r -> b 1 n r")
        latent = self.autoencoder.bottleneck.from_ids(indices)
        return self.autoencoder.decode(latent)

    def forward(self, x: Tensor, texts: List[str]) -> Tensor:
        indices = self.encode_audio(x)
        embedding = self.encode_text(texts)
        return self.transformer_decoder(indices, context=embedding)

    def training_step(self, batch, batch_idx):
        waveforms, info = batch
        loss = self(waveforms, texts=info["text"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, info = batch
        loss = self(waveforms, texts=info["text"])
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.transformer_encoder.parameters()},
                {"params": self.transformer_decoder.parameters()},
            ],
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    @property
    def device(self):
        return next(self.transformer_decoder.parameters()).device


""" Datamodule """


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        *,
        val_split: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Any = None
        self.data_val: Any = None

    def setup(self, stage: Any = None) -> None:
        split = [1.0 - self.val_split, self.val_split]
        self.data_train, self.data_val = fractional_random_split(self.dataset, split)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    print("WandbLogger not found.")
    return None


def log_wandb_audio_batch(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
    logger.log(
        {
            f"sample_{idx}_{id}": wandb.Audio(
                samples[idx],
                caption=caption,
                sample_rate=sampling_rate,
            )
            for idx in range(num_items)
        }
    )


def log_wandb_audio_spectrogram(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = samples.detach().cpu()
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
        center=True,
        norm="slaney",
    )

    def get_spectrogram_image(x):
        spectrogram = transform(x[0])
        image = librosa.power_to_db(spectrogram)
        trace = [go.Heatmap(z=image, colorscale="viridis")]
        layout = go.Layout(
            yaxis=dict(title="Mel Bin (Log Frequency)"),
            xaxis=dict(title="Frame"),
            title_text=caption,
            title_font_size=10,
        )
        fig = go.Figure(data=trace, layout=layout)
        return fig

    logger.log(
        {
            f"mel_spectrogram_{idx}_{id}": get_spectrogram_image(samples[idx])
            for idx in range(num_items)
        }
    )


class SampleLogger(Callback):
    def __init__(
        self, num_items: int, channels: int, sampling_rate: int, length: int
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.epoch_count = 0

        self.log_next = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment

        # Encode x_true to get indices
        waveform, info = batch
        waveform = waveform[0 : self.num_items]

        log_wandb_audio_batch(
            logger=wandb_logger,
            id="true",
            samples=waveform,
            sampling_rate=self.sampling_rate,
        )
        log_wandb_audio_spectrogram(
            logger=wandb_logger,
            id="true",
            samples=waveform,
            sampling_rate=self.sampling_rate,
        )

        texts = info["text"][0 : self.num_items]

        indices = pl_module.encode_audio(waveform)
        embedding = pl_module.encode_text(texts)

        length = indices.shape[1]
        indices_generated = pl_module.transformer_decoder.generate(
            start_tokens=indices[:, 0:1],  # Provide first two time tokens
            sequence_length=length,  # Autoregressively predict another full sequence
            context=embedding,
            keep_start=True,
        )

        print(indices[:, 0:1].shape, length, indices.shape, indices_generated.shape)

        samples = pl_module.decode_audio(indices_generated)

        log_wandb_audio_batch(
            logger=wandb_logger,
            id="recon",
            samples=samples,
            sampling_rate=self.sampling_rate,
        )
        log_wandb_audio_spectrogram(
            logger=wandb_logger,
            id="recon",
            samples=samples,
            sampling_rate=self.sampling_rate,
        )

        if is_train:
            pl_module.train()
