import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
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
num_images = st.slider('Number of Images to Generate:', min_value=1, max_value=50, value=20)

# Button to generate images
if st.button('Generate Images'):
    # Generate random latent vectors (noise)
    z = torch.randn(num_images, 100, 1, 1, device=device)

    # Generate images from the noise vectors
    with torch.no_grad():
        generated_images = generator(z)

    # Convert the tensor to a grid and denormalize it
    grid = make_grid(generated_images, nrow=5, normalize=True)
    np_img = grid.cpu().numpy()

    # Display the generated images using Matplotlib
    st.pyplot(plt.figure(figsize=(6, 6)))
    plt.axis('off')
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
