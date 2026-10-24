import os
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


def get_num_workers():
    """
    Détecte automatiquement le nombre de workers à utiliser
    selon le nombre de cœurs du CPU.
    - Windows : max 2 workers (spawn coûteux)
    - Linux/Mac : max 4 workers
    - 1 cœur : 0 worker (obligatoire)
    """
    nb_coeurs = os.cpu_count() or 1
    
    if nb_coeurs <= 1:
        return 0
    
    # Sur Windows, les workers sont plus coûteux (spawn)
    max_workers = 2 if platform.system() == "Windows" else 4
    
    return min(nb_coeurs - 1, max_workers)


def main():
    # =====================================================
    # CONFIGURATION
    # =====================================================
    DATA_DIR = "data_set_vegetation/seg_train/seg_train"
    MODEL_PATH = "vegetation_model.pth"
    NUM_EPOCHS = 8
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    TAILLE_IMAGE = 224

    # ✅ Détection automatique du nombre de workers
    NUM_WORKERS = get_num_workers()

    print("=" * 60)
    print(f"PC détecté : {platform.system()} / {platform.machine()}")
    print(f"CPU : {os.cpu_count()} cœurs disponibles")
    print(f"num_workers utilisé : {NUM_WORKERS}")
    print("=" * 60)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(TAILLE_IMAGE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # =====================================================
    # CHARGEMENT DES DONNÉES
    # =====================================================
    print("Chargement des images...")
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    print(f"Classes détectées : {dataset.classes}")
    print(f"Total images : {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # =====================================================
    # MODÈLE
    # =====================================================
    print("Chargement de ResNet50 pré-entraîné...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_classes = len(dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # =====================================================
    # ENTRAÎNEMENT
    # =====================================================
    print("\nDébut de l'entraînement...")
    best_val_acc = 0

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
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

        # --- Val ---
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_size
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': dataset.classes,
                'num_classes': num_classes,
            }, MODEL_PATH)
            print(f"  💾 Nouveau meilleur modèle sauvegardé ({val_acc:.2f}%)")

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} "
            f"| Train: {train_acc:.2f}% | Val: {val_acc:.2f}%"
        )

    print(f"\n✅ Entraînement terminé. Meilleure précision : {best_val_acc:.2f}%")
    print(f"Classes : {dataset.classes}")


# ✅ PROTECTION OBLIGATOIRE SUR WINDOWS
if __name__ == '__main__':
    main()