import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
# 1. Define project paths
PROJECT_DIR = Path(r"D:\TB_Project")
CSV_PATH = PROJECT_DIR / "data" / "dataset.csv"
# 2. Load the dataset
df = pd.read_csv(CSV_PATH)
# Exclude the one corrupted image from training and evaluation
bad_image_name = "CHNCXR_0338_1.png"

df = df[
    ~df["image"].str.endswith(bad_image_name)
].copy()

print("Images after excluding corrupted file:", len(df))
# Keep only the columns needed for image classification
df = df[["image", "label"]].copy()

# Split into training (70%), validation (15%), and test (15%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)
# Display the results
print("Total images:", len(df))
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

print("\nTraining label counts:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation label counts:")
print(val_df["label"].value_counts().sort_index())

print("\nTest label counts:")
print(test_df["label"].value_counts().sort_index())


from PIL import Image
from torchvision import transforms

# Prepare every image for the model
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load one example image
sample_path = train_df.iloc[0]["image"]
sample_image = Image.open(sample_path).convert("RGB")
sample_tensor = image_transform(sample_image)

# Display the result
print("\nOriginal image size:", sample_image.size)
print("Prepared image shape:", sample_tensor.shape)
print("Pixel value range:", sample_tensor.min().item(),
      "to", sample_tensor.max().item())

from torch.utils.data import Dataset, DataLoader

# Create a dataset that loads X-ray images and labels
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe.loc[index, "image"]
        label = self.dataframe.loc[index, "label"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(label)


# Create datasets for all three groups
train_dataset = ChestXrayDataset(train_df, image_transform)
val_dataset = ChestXrayDataset(val_df, image_transform)
test_dataset = ChestXrayDataset(test_df, image_transform)

# Create loaders that supply images in batches
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Check one batch
images, labels = next(iter(train_loader))

print("\nTraining batch image shape:", images.shape)
print("Training batch label shape:", labels.shape)
print("Labels in this batch:", labels.tolist())

import torch
import torch.nn as nn
from torchvision import models

# Use CPU for training
device = torch.device("cpu")

# Load a ResNet-18 model with pretrained image features
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Replace the final layer for our two classes
model.fc = nn.Linear(model.fc.in_features, 2)

# Move the model to the selected device
model = model.to(device)

print("\nModel:", model.__class__.__name__)
print("Device:", device)
print("Output classes:", model.fc.out_features)

import time
import matplotlib.pyplot as plt

# Freeze the pretrained layers; train only the final layer
for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in model.fc.parameters():
    parameter.requires_grad = True

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Store loss values for plotting
train_losses = []
val_losses = []

epochs = 3
start_time = time.time()

for epoch in range(epochs):
    # Training phase
    model.eval()
    model.fc.train()

    running_train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)

    epoch_train_loss = running_train_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)

    # Validation phase
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)

    epoch_val_loss = running_val_loss / len(val_dataset)
    val_losses.append(epoch_val_loss)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Training Loss: {epoch_train_loss:.4f} | "
        f"Validation Loss: {epoch_val_loss:.4f}"
    )

# Plot and save the loss curves
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker="o",
         label="Training Loss")
plt.plot(range(1, epochs + 1), val_losses, marker="o",
         label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet-18 Training and Validation Loss")
plt.xticks(range(1, epochs + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = PROJECT_DIR / "loss_curves.png"
plt.savefig(plot_path, dpi=150)
plt.close()

# Save the trained model
model_path = PROJECT_DIR / "resnet18_image_model.pth"
torch.save(model.state_dict(), model_path)

print(f"\nLoss plot saved to: {plot_path}")
print(f"Model saved to: {model_path}")
print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")

