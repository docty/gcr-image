import streamlit as st 
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision.utils import save_image
from io import BytesIO


st.title("DCGAN")
st.write(
    "We are about to deploy a quick pytorch project"
)

# Load your DCGAN model (Generator)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.conv_blocks(z)
        return img

# Load the trained generator model (weights should be saved)
def load_generator():
    generator = Generator()
    # Assuming the model weights have been saved as 'generator.pth'
    generator.load_state_dict(torch.load("generator.pth"))
    generator.eval()
    return generator

# Generate image function
def generate_image(generator):
    # Generate random noise as input to the generator
    z = torch.randn(1, 100, 1, 1)  # Batch size = 1, Latent vector size = 100
    with torch.no_grad():
        gen_img = generator(z).detach().cpu()
    
    # Convert the image to a format that can be displayed in Streamlit
    gen_img = gen_img.squeeze().permute(1, 2, 0).numpy()
    gen_img = (gen_img + 1) / 2  # Denormalize to [0, 1]
    
    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(gen_img)
    ax.axis('off')
    
    # Convert the Matplotlib figure to an image in memory
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Streamlit App UI
def main():
    # Streamlit App title
    st.title("DCGAN Plant Image Generator")
    
    # Load the model
    generator = load_generator()
    
    # Add a button to generate images
    if st.button("Generate Plant Image"):
        image_buffer = generate_image(generator)
        
        # Display the generated image
        st.image(image_buffer, caption="Generated Plant Image", use_column_width=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
