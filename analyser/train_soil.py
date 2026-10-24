import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

DATA_DIR = "data_set"
MODEL_PATH = "soil_model.pth"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Chargement des images...")
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print(f"Classes détectées : {dataset.classes}")
print(f"Total images : {len(dataset)}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print("Chargement de ResNet50 pré-entraîné...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nDébut de l'entraînement...")
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    train_correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_size
    avg_loss = train_loss / len(train_loader)

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_size
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

torch.save({
    'model_state_dict': model.state_dict(),
    'classes': dataset.classes,
    'num_classes': num_classes,
}, MODEL_PATH)
print(f"\n✅ Modèle sauvegardé dans {MODEL_PATH}")
print(f"Classes : {dataset.classes}")