import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Ayarlar
DATA_DIR = 'data'
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 20
VALIDATION_SPLIT = 0.2
MODEL_PATH = 'recycle_model.pt'
NUM_CLASSES = 6

# Veri artırma ve ön işleme
train_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset ve DataLoader
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
class_names = full_dataset.classes
num_total = len(full_dataset)
num_val = int(num_total * VALIDATION_SPLIT)
num_train = num_total - num_val
train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])
# Doğrulama için augmentation olmadan tekrar transform ayarla
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Model
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# Cihaz seçimi
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Eğitim döngüsü
def train():
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / num_train
        epoch_acc = running_corrects.double() / num_train
        print(f"[Train] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Doğrulama
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        val_loss = val_loss / num_val
        val_acc = val_corrects.double() / num_val
        print(f"[Val]   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Yeni en iyi model kaydedildi: {MODEL_PATH}")

if __name__ == '__main__':
    print(f"Sınıflar: {class_names}")
    train()
    print(f"Eğitim tamamlandı. En iyi model '{MODEL_PATH}' olarak kaydedildi.") 