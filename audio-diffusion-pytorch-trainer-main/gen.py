from main import module_base
from audio_diffusion_pytorch import AudioDiffusionModel, UniformDistribution

# First create the AudioDiffusionModel instance with your config parameters
audio_diffusion_model = AudioDiffusionModel(
    in_channels=2,  # from your channels config
    channels=128,
    patch_size=16,
    resnet_groups=8,
    kernel_multiplier_downsample=2,
    multipliers=[1, 2, 4, 4, 4, 4, 4],
    factors=[4, 4, 4, 2, 2, 2],
    num_blocks=[2, 2, 2, 2, 2, 2],
    attentions=[0, 0, 0, 1, 1, 1, 1],
    attention_heads=8,
    attention_features=64,
    attention_multiplier=2,
    use_nearest_upsample=False,
    use_skip_scale=True,
    diffusion_sigma_distribution=UniformDistribution()
)

# Then load the checkpoint with the audio diffusion model instance
model = module_base.Model.load_from_checkpoint(
    checkpoint_path='logs/ckpts/2024-10-31-17-42-45/epoch=716-valid_loss=0.009.ckpt',
    lr=1e-4,
    lr_beta1=0.95,
    lr_beta2=0.999,
    lr_eps=1e-6,
    lr_weight_decay=1e-3,
    model=audio_diffusion_model,
    ema_beta=0.9999,
    ema_power=0.7
)
# Generate a sample
import torch
from audio_diffusion_pytorch import VSampler, LinearSchedule
import torchaudio
import torch
import torchaudio
from audio_diffusion_pytorch import VSampler, LinearSchedule



@torch.no_grad()
def generate_audio_with_params(
        model,
        num_samples=1,
        length=524288,  # Keep this fixed to match training
        num_steps=3,
        channels=2,
        sampling_rate=48000,
        device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()
    diffusion_model = model.model_ema.ema_model.to(device)

    # Ensure length is divisible by model's patch_size (16 in your case)
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
# Example usage:
try:
    # Generate a short sample first to test
    samples, sr = generate_audio_with_params(
        model,
        num_samples=1,
        num_steps=500
    )

    # Save the test sample
    audio = samples[0].cpu()
    audio = audio / torch.abs(audio).max()
    torchaudio.save(
        'test_generated_audio.wav',
        audio,
        sr,
        format='wav'
    )

except Exception as e:
    print(f"Error occurred: {str(e)}")