#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import logging
import optuna
import multiprocessing

# Import model-specific modules from ncps.
from ncps.wirings import AutoNCP
from ncps.torch import CfC

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global device configuration: use CUDA if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################
# Cell 2: DrivingDataset Class
##############################

# Define CSV column names.
COLUMN_NAMES = ["center", "left", "right", "steering", "throttle", "brake", "speed"]

class DrivingDataset(Dataset):
    """Dataset for autonomous driving images and targets."""
    def __init__(self, csv_file, root_dir, transform=None, sequence_length=5):
        """
        Args:
            csv_file (str): Path to the CSV file.
            root_dir (str): Directory where images are stored.
            transform (callable, optional): Transformations to apply to images.
            sequence_length (int): Number of consecutive frames per sample.
        """
        self.df = pd.read_csv(csv_file, names=COLUMN_NAMES)
        # Keep only required columns.
        self.df = self.df[["center", "left", "right", "steering", "throttle", "brake"]]
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.sequence_length = sequence_length

    def get_image_path(self, col_value):
        """Construct full image path from CSV entry."""
        filename = Path(col_value.strip()).name
        # Assumes images are stored in "IMG" subfolder under the root_dir.
        return self.root_dir / "IMG" / filename

    def load_image(self, path: Path):
        """Load image and convert to RGB. On failure, returns a blank image."""
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            # Return a blank image with a default size (200x66); adjust as needed.
            img = Image.new("RGB", (200, 66))
        return img

    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.df) - self.sequence_length + 1

    def __getitem__(self, idx):
        images_seq = []
        # Iterate over the sequence of frames.
        for i in range(self.sequence_length):
            row = self.df.iloc[idx + i]
            # Get paths for center, left, and right images.
            center_path = self.get_image_path(row["center"])
            left_path = self.get_image_path(row["left"])
            right_path = self.get_image_path(row["right"])
            
            # Load images with error handling.
            center_img = self.load_image(center_path)
            left_img = self.load_image(left_path)
            right_img = self.load_image(right_path)
            
            # Apply transformations if provided.
            if self.transform:
                center_img = self.transform(center_img)
                left_img = self.transform(left_img)
                right_img = self.transform(right_img)
            
            # Stack camera views → shape: (3, channels, height, width)
            images = torch.stack([center_img, left_img, right_img], dim=0)
            images_seq.append(images)
        
        # Stack sequence → shape: (sequence_length, 3, channels, height, width)
        images_seq = torch.stack(images_seq, dim=0)
        
        # Use target values from the last frame.
        target_row = self.df.iloc[idx + self.sequence_length - 1]
        target = torch.tensor([
            target_row["steering"],
            target_row["throttle"],
            target_row["brake"]
        ], dtype=torch.float32)
        return images_seq, target

##############################
# Cell 3: Data Transformations and Initialization
##############################

# Define image transformations.
transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Update these paths to match your dataset location.
csv_path = r"C:\Users\harsh\OneDrive\Desktop\Udacity datset 2\self_driving_car_dataset_make\driving_log.csv"
root_dir = r"C:\Users\harsh\OneDrive\Desktop\Udacity datset 2\self_driving_car_dataset_make"
sequence_length = 5

# Create a dataset instance.
dataset = DrivingDataset(csv_file=csv_path, root_dir=root_dir, transform=transform, sequence_length=sequence_length)
print("Total samples in dataset:", len(dataset))
sample_images, sample_target = dataset[0]
print("Sample images shape:", sample_images.shape)  # Expected: (sequence_length, 3, channels, 66, 200)
print("Sample target (steering, throttle, brake):", sample_target)

##############################
# Cell 4: TemporalSequenceLearner Model Definition
##############################

class TemporalSequenceLearner(nn.Module):
    def __init__(self, hidden_neurons, image_channels=3, pretrained_weights_path=None):
        """
        Args:
            hidden_neurons (int): Number of hidden units for the CfC wiring.
            image_channels (int): Number of input image channels (e.g., 3 for RGB).
            pretrained_weights_path (str, optional): Path to pretrained weights.
        """
        super().__init__()
        # Shared feature extractor for individual images.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(image_channels, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        if pretrained_weights_path:
            self.feature_extractor.load_state_dict(torch.load(pretrained_weights_path))
        # Freeze feature extractor parameters.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Computed feature dimension for images of size (3, 66, 200).
        self.feature_dim = 1152
        
        # Configure AutoNCP to output 3 values (steering, throttle, brake).
        wiring = AutoNCP(hidden_neurons,3,sparsity_level=0.75)
        # Convert wiring matrices to CPU tensors (avoid GPU-to-NumPy conversion issues).
        wiring.adjacency_matrix = torch.tensor(wiring.adjacency_matrix).cpu()
        if wiring.sensory_adjacency_matrix is not None:
            wiring.sensory_adjacency_matrix = torch.tensor(wiring.sensory_adjacency_matrix).cpu()
        
        self.classifier = CfC(self.feature_dim, wiring)
        # Final output layer maps CfC output to a 3-element vector.
        self.output_layer = nn.Linear(3, 3)
    
    def forward(self, images):
        """
        Args:
            images: Tensor of shape (batch, T, 3, channels, height, width).
        Returns:
            Tensor of shape (batch, 3) representing steering, throttle, and brake.
        """
        batch, T, num_views, channels, height, width = images.size()
        # Merge batch, time, and view dimensions.
        images = images.view(batch * T * num_views, channels, height, width)
        features = self.feature_extractor(images)
        features = features.view(features.size(0), -1)  # Flatten features.
        # Reshape to (batch, T, num_views, feature_dim) and average over views.
        features = features.view(batch, T, num_views, self.feature_dim).mean(dim=2)
        # Process temporal sequence with CfC; output shape: (batch, T, 3).
        classifier_out, _ = self.classifier(features)
        final_time_step = classifier_out[:, -1, :]  # Use last time step.
        x = self.output_layer(final_time_step)       # (batch, 3)
        # Apply activation constraints: steering via tanh, throttle and brake via sigmoid.
        steering = torch.tanh(x[:, 0:1])
        throttle = torch.sigmoid(x[:, 1:2])
        brake = torch.sigmoid(x[:, 2:3])
        return torch.cat([steering, throttle, brake], dim=1)

##############################
# Cell 5: Data Preparation & Hyperparameter Tuning
##############################

def prepare_data(sequence_length=5, batch_size=512):
    """
    Prepares training and validation data loaders.
    """
    csv_path = r"C:\Users\harsh\OneDrive\Desktop\Udacity datset 2\self_driving_car_dataset_make\driving_log.csv"
    root_dir = r"C:\Users\harsh\OneDrive\Desktop\Udacity datset 2\self_driving_car_dataset_make"
    transform = transforms.Compose([
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = DrivingDataset(csv_file=csv_path, root_dir=root_dir, transform=transform, sequence_length=sequence_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def objective(trial):
    """
    Objective function for hyperparameter tuning with Optuna.
    """
    hidden_neurons = trial.suggest_int("hidden_neurons", 16, 256, step=16)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    train_loader, val_loader = prepare_data(sequence_length)
    model = TemporalSequenceLearner(hidden_neurons)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    num_epochs = 5  # Short training for tuning.
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, target in train_loader:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        trial.report(epoch_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # Evaluate on validation set.
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.to(device), target.to(device)
            outputs = model(images)
            loss = criterion(outputs, target)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss

def run_optuna_study(n_trials=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (validation loss): {trial.value}")
    print("  Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

##############################
# Cell 6: Final Training & Model Saving
##############################

def final_train(model, train_loader, criterion, optimizer, device, num_epochs):
    """
    Trains the final model.
    """
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, target in train_loader:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Final Training Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

def main():
    # Uncomment the next line to run hyperparameter tuning.
    # run_optuna_study(n_trials=50)
    
    # Prepare data loaders.
    train_loader, _ = prepare_data(sequence_length)
    
    # Final training with chosen hyperparameter (example: hidden_neurons=19).
    final_model = TemporalSequenceLearner(hidden_neurons=36)
    final_model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.1)
    final_epochs = 20
    
    final_train(final_model, train_loader, criterion, optimizer, device, final_epochs)
    
    # Save the final model.
    torch.save(final_model.state_dict(), "final_temporal_sequence_model.pth")
    print("Final model saved as 'final_temporal_sequence_model.pth'.")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Needed for Windows.
    main()
