import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

# -------------------------------
# Custom Dataset
# -------------------------------
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform=None, image_col='id_code', label_col='label', img_ext='.png'):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.img_ext = img_ext

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx][self.image_col] + self.img_ext
        img_path = os.path.join(self.img_dir, img_name)
        label = int(self.df.iloc[idx][self.label_col])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# -------------------------------
# Training Loop
# -------------------------------
def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, save_path):
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        val_loss /= len(val_loader)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # -------------------------------
        # Save Best Model
        # -------------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved with Val Acc: {val_acc:.2f}%")

        scheduler.step()

    print(f"Training completed. Best Val Accuracy: {best_acc:.2f}%")


# -------------------------------
# Main Function
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-csv', type=str, required=True, help='Path to CSV with labels')
    parser.add_argument('--img-dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--save-path', type=str, default="ml/models/best_model.pth", help='Path to save best model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Load CSV and rename diagnosis to label
    df = pd.read_csv(args.labels_csv)
    if 'diagnosis' in df.columns:
        df = df.rename(columns={'diagnosis': 'label'})

    print("CSV columns:", df.columns.tolist())

    # Train-validation split with stratification
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Data Augmentation and Normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dataset and DataLoader
    train_dataset = CustomImageDataset(train_df, args.img_dir, transform=train_transform, image_col='id_code', label_col='label')
    val_dataset = CustomImageDataset(val_df, args.img_dir, transform=val_transform, image_col='id_code', label_col='label')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model setup - pretrained ResNet50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(df['label'].unique()))
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Start training
    train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs, args.save_path)


if __name__ == '__main__':
    main()
