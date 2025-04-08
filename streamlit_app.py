import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import streamlit as st

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Load model and define the device
generator_checkpoint_path = "generator.pth"  # Replace with the correct checkpoint file path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator()
generator.load_state_dict(torch.load(generator_checkpoint_path, weights_only=True, map_location=device))
generator.eval()  # Set the model to evaluation mode
generator.to(device)

# Streamlit UI elements
st.title('Generative Model Image Generator')
num_images = 5  # Display exactly 5 images

# Button to generate images
if st.button('Generate 5 Images'):
    # Generate random latent vectors (noise)
    z = torch.randn(num_images, 100, 1, 1, device=device)

    # Generate images from the noise vectors
    with torch.no_grad():
        generated_images = generator(z)

    # Create columns to display images in a row
    columns = st.columns(num_images)

    # Display each image in a separate column
    for i in range(num_images):
        np_img = generated_images[i].cpu().numpy()

        # Rescale the image from [-1, 1] to [0, 255] and transpose the dimensions to (H, W, C)
        np_img = (np_img + 1) / 2  # Rescale to [0, 1]
        np_img = np.transpose(np_img, (1, 2, 0))  # Change from CxHxW to HxWxC

        # Display the individual image using st.image in the corresponding column
        columns[i].image(np_img, caption=f'Generated Image {i+1}', use_column_width=True)
