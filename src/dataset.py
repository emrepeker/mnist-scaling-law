# MNIST dataset + subsampling functions]

# After You run this file you get sample under /data file
# label, img(1x784) -> transformed [0,1]

import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import pandas as pd
import os 
# Make sure there is data file
os.makedirs("data", exist_ok=True)

# Transform: just convert to tensor (normalize manually)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST train set (60k sample)
train_dataset = datasets.MNIST(
    root="./data",
    train=True, 
    download=True,
    transform=transform
)

# Subset sizes
subset_sizes = [1000, 5000, 10000, 20000, 40000, 60000]

# Fixed Generator for reproducibility 
generator = torch.Generator().manual_seed(31)

# Create and save CSV subsets
for size in subset_sizes:
    subset,_= random_split(train_dataset,
                           [size , len(train_dataset)- size] ,
                           generator=generator)
    
    rows = []
    for i in range(len(subset)):
        img, label = subset[i]
        img = img.view(-1).numpy() # Flatten 28x28 -> 784
        rows.append([label]+ img.tolist()) # [0,1] since ToTensor divides by 255
    df = pd.DataFrame(rows)
    df.to_csv(f"data/mnist_train_{size}.csv", index = False)
    print(f"Saved data/mnist_train_{size}.csv with {size} normalized samples")