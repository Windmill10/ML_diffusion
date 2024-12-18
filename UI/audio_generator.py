# audio_generator.py
import numpy as np
from scipy.io.wavfile import write
import os
import sys
import importlib
import audio_diffusion_pytorch
importlib.reload(audio_diffusion_pytorch)


sys.path.append("../audio_diffusion_pytorch_trainer_main")

from main import module_base
from audio_diffusion_pytorch import AudioDiffusionModel, UniformDistribution, VSampler, LinearSchedule



def generate_audio(species: str, seed):
    # Create the DiffusionModel instance with your config parameters
    audio_diffusion_model = AudioDiffusionModel(
        in_channels=2,  # from your channels config
        channels=[64],  # Example channels configuration
        resnet_groups=8,
        kernel_multiplier_downsample=2,
        multipliers=[4, 2, 2, 3, 3, 3, 3, 3],  # [1, 2, 4, 4, 4, 4, 4]
        factors=[2, 2, 2, 2, 2, 2, 2],  # [4, 4, 4, 2, 2, 2]
        num_blocks=[2, 2, 2, 2, 4, 4, 4],  # [2, 2, 2, 2, 2, 2]
        attentions=[0, 0, 0, 0, 1, 2, 2],  # [0, 0, 0, 1, 1, 1, 1]
        attention_heads=8,
        attention_features=64,
        attention_multiplier=4,  # 2
        use_nearest_upsample=False,
        use_skip_scale=True,
        diffusion_sigma_distribution=UniformDistribution(),
        patch_factor=1,
        patch_blocks=1
    )
    print("Diffusion Model created successfully.")

    # Load the checkpoint with the audio diffusion model instance
    model = module_base.Model.load_from_checkpoint(
        checkpoint_path=f'ckpts/{species}.ckpt',

        lr=1e-4,
        lr_beta1=0.95,
        lr_beta2=0.999,
        lr_eps=1e-6,
        lr_weight_decay=1e-3,
        model=audio_diffusion_model,
        ema_beta=0.995,
        ema_power=0.7
    )
    print("Model loaded successfully.")

    # Generate a sample
    import torch
    import torchaudio

    @torch.no_grad()
    def generate_audio_with_params(
            model,
            num_samples=1,
            length=80000,  # Match the length from your configuration
            num_steps=20,
            channels=2,
            sampling_rate=16000,
            device=None
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = model.to(device)
        model.eval()
        diffusion_model = model.model_ema.ema_model.to(device)

        # Ensure length is divisible by model's patch_size
        patch_size = 16
        length = (length // patch_size) * patch_size

        # Create noise input
        noise = torch.randn((num_samples, channels, length), device=device)

        # Setup sampler and schedule
        sampler = VSampler()
        schedule = LinearSchedule()

        # Generate samples
        samples = diffusion_model.sample(
            noise=noise,
            sampler=sampler,
            sigma_schedule=schedule,
            num_steps=num_steps
        )

        return samples, sampling_rate

    samples, sr = generate_audio_with_params(
        model,
        num_samples=1,
        num_steps=20,
        length=80000,
        sampling_rate=16000,
        channels=2
    )

    # Save the generated audio
    audio = samples[0].cpu()
    audio = audio / torch.abs(audio).max()
    torchaudio.save(
        'generated.wav',
        audio,
        sr,
        format='wav'
    )
    print("Audio generated and saved successfully.")