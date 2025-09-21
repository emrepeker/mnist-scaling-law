# Simple MLP / CNN architectures
# Importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Creating datasets
train_data=torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
test_data=torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Splitting datasets into mini-batches
def prepare_data(batch_size):
  train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
  return train_dataloader, test_dataloader

# Defining the Neural Network Architecture
def create_model(input_size, hidden_layers, output_size):
  class FNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
      super(FNN, self).__init__()
      # Creating layers
      self.layers = nn.ModuleList()
      for hidden_size in hidden_layers:
        self.layers.append(nn.Linear(input_size, hidden_size))
        input_size = hidden_size
      self.out = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = nn.ReLU()(layer(x))
        x = self.out(x)
        return x

  model = FNN(input_size, hidden_layers, output_size)
  return model

# Training loop
def train_model(model, train_dataloader, epochs, learning_rate):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(),lr=learning_rate)
  train_losses=[]
  for epoch in range(epochs):
    running_loss=0
    for inputs, labels in train_dataloader:
      outputs = model(inputs)
      loss= criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
    running_loss/=len(train_dataloader)
    train_losses.append(running_loss)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}')
  # Visualizing Results
  plt.plot(train_losses)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training Loss over Epochs")
  plt.show()

  # Evaluation on Test Data

def evaluate_model(model, test_dataloader):
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f'Test Accuracy: {accuracy:.2f}%')
  return accuracy

def run_experiment(batch_size=64, epochs=5, learning_rate=0.01, hidden_layers=[128,64,32]):
  # 1. Data
  train_loader, test_loader = prepare_data(batch_size)
  # 2. Model
  model = create_model(input_size=28*28, hidden_layers=hidden_layers, output_size=10)
  # 3. Train
  train_model(model, train_loader, epochs, learning_rate)
  # 4. Evaluate
  evaluate_model(model, test_loader)
  return model



# Testing
experiments = [
    {"hidden_layers": [64], "batch_size": 64, "epochs": 5, "learning_rate": 0.01},
    {"hidden_layers": [128], "batch_size": 64, "epochs": 5, "learning_rate": 0.01},
    {"hidden_layers": [128, 64], "batch_size": 64, "epochs": 5, "learning_rate": 0.01},
    {"hidden_layers": [256, 128, 64], "batch_size": 128, "epochs": 10, "learning_rate": 0.005},
    {"hidden_layers": [512, 256, 128, 64], "batch_size": 128, "epochs": 10, "learning_rate": 0.005},
]

for i, exp in enumerate(experiments):
    print(f"\n--- Running Experiment {i+1} ---")
    run_experiment(
        batch_size=exp["batch_size"],
        epochs=exp["epochs"],
        learning_rate=exp["learning_rate"],
        hidden_layers=exp["hidden_layers"]
    )
