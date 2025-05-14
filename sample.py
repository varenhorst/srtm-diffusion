import torch
from diffusers import UNet2DModel, DDPMScheduler
import torchvision.utils as vutils
import os
import imageio
from torchvision.transforms.functional import to_pil_image

# Set up paths
model_dir = "grayscale_diffusion_model"
output_dir = "samples"
os.makedirs(output_dir, exist_ok=True)

# Load model and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet2DModel.from_pretrained(model_dir).to(device)
scheduler = DDPMScheduler.from_pretrained(model_dir)

@torch.no_grad()
def generate_images(model, scheduler, num_samples=1, image_size=128):
    model.eval()
    # Start with random noise
    x = torch.randn((num_samples, 1, image_size, image_size)).to(device)

    for t in reversed(range(scheduler.config.num_train_timesteps)):
        timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x, timesteps).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    # Rescale from [-1,1] to [0,1]
    x = (x.clamp(-1, 1) + 1) / 2
    return x


@torch.no_grad()
def generate_images_gif(model, scheduler, num_samples=1, image_size=128, save_gif_path="diffusion.gif"):
    model.eval()
    x = torch.randn((num_samples, 1, image_size, image_size)).to(device)
    frames = []

    for t in reversed(range(scheduler.config.num_train_timesteps)):
        timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x, timesteps).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

        # Prepare the current frame (convert from [-1, 1] to [0, 255])
        x_frame = (x[0].clamp(-1, 1) + 1) / 2  # Get the first sample
        img = to_pil_image(x_frame.cpu())
        frames.append(img)

    # Save as GIF
    frames[0].save(
        save_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=40,  # ms per frame
        loop=0
    )
    print(f"GIF saved to {save_gif_path}")


# Generate and save samples
print("Generating samples...")
samples = generate_images(model, scheduler, num_samples=1)
vutils.save_image(samples, os.path.join(output_dir, "generated_heightmaps.png"), nrow=4)
print("Samples saved to:", os.path.join(output_dir, "generated_heightmaps.png"))


print("Generting Generation GIF")
generate_images_gif(model, scheduler, num_samples=1, save_gif_path="samples/diffusion_process.gif")

