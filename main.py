print("Script started")
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 1. Define transforms (resize + normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # ResNet/EfficientNet expect this size
    transforms.ToTensor(),               # convert PIL image -> tensor
    transforms.Normalize(                # normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 2. Load the full dataset from the folder structure
dataset = datasets.ImageFolder(root="data", transform=transform)

# Check what classes it found
print("Classes found:", dataset.classes)
print("Total images:", len(dataset))

# 3. Split into train/val (e.g. 80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 4. Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
print(dataset.classes)
print([dataset.targets.count(i) for i in range(len(dataset.classes))])
import torch.nn as nn
from torchvision import models

# 1. Load a pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. Freeze all existing layers (don't retrain what it already knows)
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the final layer to output 2 classes instead of 1000
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# 4. Move model to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Model ready. Using device:", device)
print(model.fc)
import torch.optim as optim

# Class weighting: penalize mistakes on the minority class (diseased) more
diseased_count, healthy_count = 382, 572
total = diseased_count + healthy_count
weight_diseased = total / (2 * diseased_count)
weight_healthy = total / (2 * healthy_count)

class_weights = torch.tensor([weight_diseased, weight_healthy]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer: updates the model's weights based on the loss
# Only optimizing model.fc.parameters() since everything else is frozen
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

def train_one_epoch():
    model.train()  # put model in training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear old gradients
        outputs = model(images)        # forward pass: get predictions
        loss = criterion(outputs, labels)  # compare to real labels
        loss.backward()                # backward pass: compute gradients
        optimizer.step()               # update weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate_one_epoch():
    model.eval()  # put model in evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # don't compute gradients, we're not training here
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# Actual training loop
num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate_one_epoch()
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)
print("\nClass order:", dataset.classes)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))
torch.save(model.state_dict(), "olive_model.pth")
print("Model saved to olive_model.pth")
