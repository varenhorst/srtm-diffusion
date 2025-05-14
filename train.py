from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os


model = UNet2DModel(
    sample_size=128,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 128),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Define transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Normalize([0.5], [0.5])  # Scales to [-1,1]
])

# Load images from a directory
def load_images(directory):
    image_paths = list(Path(directory).glob("*.png"))
    images = []
    for path in image_paths:
        image = Image.open(path)
        image = transform(image)
        images.append({"image": image})
    return images

# Example usage
dataset = load_images("srtm_tiles")
# Create DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print('Training')
# Training loop
epochs = 10
for epoch in range(epochs):
    for batch in dataloader:
        clean_images = batch["image"].to(device)
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.size(0),), device=device).long()

        # Add noise to the images
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Predict the noise
        noise_pred = model(noisy_images, timesteps).sample

        # Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} completed. Loss: {loss.item()}")

# Save UNet model
model.save_pretrained("grayscale_diffusion_model")
# Save scheduler (optional, but recommended for consistent sampling)
noise_scheduler.save_pretrained("grayscale_diffusion_model")


