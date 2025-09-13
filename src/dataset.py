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
# Fixed Generator for reproducibility
generator = torch.Generator().manual_seed(31)

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

train_size = 50_000
# Split train test 50k 10k
train_set, test_set = random_split(train_dataset,
                                   [train_size, len(train_dataset) - train_size],
                                   generator=generator) 




# Get Test.csv
rows = []
for i in range(len(train_dataset) - train_size):
    img, label = test_set[i]
    img = img.view(-1).numpy() # Flatten 28x28 -> 784
    rows.append([label]+ img.tolist()) # [0,1] since ToTensor divides by 255
df = pd.DataFrame(rows)
df.to_csv(f"data/mnist_test_{len(train_dataset) - train_size}.csv", index = False)

# Subset sizes
subset_sizes = [1000, 5000, 10000, 20000, 40000, 50000]
# Create and save CSV train subsets
for size in subset_sizes:
    subset,_= random_split(train_set,
                           [size , len(train_set)- size] ,
                           generator=generator)
    
    rows = []
    for i in range(len(subset)):
        img, label = subset[i]
        img = img.view(-1).numpy() # Flatten 28x28 -> 784
        rows.append([label]+ img.tolist()) # [0,1] since ToTensor divides by 255
    df = pd.DataFrame(rows)
    df.to_csv(f"data/mnist_train_{size}.csv", index = False)
    print(f"Saved data/mnist_train_{size}.csv with {size} normalized samples")