import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import timm

from ml.utils import RetinopathyDataset, get_transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_model(model_name='resnet50', n_classes=5, pretrained=True):
    """
    Create a model for Diabetic Retinopathy classification

    Args:
        model_name: Model architecture (default: 'resnet50' as specified in final year report)
        n_classes: Number of DR severity classes (default: 5)
        pretrained: Use ImageNet pretrained weights (default: True)

    Returns:
        PyTorch model

    Note: Using ResNet50 architecture for consistency with final year report documentation
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=n_classes)
    return model

def compute_class_weights(df, label_col='label'):
    counts = df[label_col].value_counts().sort_index().values
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float)

def train_loop(
    labels_df,
    img_dir,
    model_name='resnet50',  # Changed to resnet50 to match final year report
    image_size=224,
    epochs=10,
    batch_size=16,
    lr=2e-4,
    device=None,
    out_dir='ml/models',
    num_workers=4,
    resume_checkpoint=None,
    early_stopping_patience=5,
):
    set_seed(42)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(out_dir, exist_ok=True)

    print("Columns in dataframe:", labels_df.columns.tolist())

    # Train/Validation split stratified by label
    train_df, val_df = train_test_split(
        labels_df, test_size=0.2, stratify=labels_df['label'], random_state=42
    )

    train_t, valid_t = get_transforms(image_size)

    train_ds = RetinopathyDataset(
        train_df, img_dir, transforms=train_t, image_column='id_code', label_column='label'
    )
    val_ds = RetinopathyDataset(
        val_df, img_dir, transforms=valid_t, image_column='id_code', label_column='label'
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    n_classes = labels_df['label'].nunique()
    model = create_model(model_name=model_name, n_classes=n_classes, pretrained=True)
    model = model.to(device)

    class_weights = compute_class_weights(labels_df, 'label').to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 1

    # Resume checkpoint if exists
    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming training from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate after scheduler step: {current_lr:.6f}")

        # Early Stopping & Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            cp_path = os.path.join(out_dir, 'best_model.pth')
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'model_name': model_name,
                    'image_size': image_size,
                    'n_classes': n_classes,
                },
                cp_path,
            )
            print(f"Saved best model to {cp_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return os.path.join(out_dir, 'best_model.pth')
