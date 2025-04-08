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
