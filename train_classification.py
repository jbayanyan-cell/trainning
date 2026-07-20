#!/usr/bin/env python3
"""
ResNet18 Classification Training Script
Called by Flask API endpoint /train/classification
"""

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from PIL import Image
from pathlib import Path
import json
from datetime import datetime

print("=" * 60, flush=True)
print("RESNET CLASSIFICATION TRAINING", flush=True)
print("=" * 60, flush=True)
sys.stdout.flush()

class ClassificationDataset(torch.utils.data.Dataset):
    """Dataset for classification with train/valid/test folders"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Get class folders
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")
        
        # Find class folders (subdirectories)
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            raise ValueError(f"No class folders found in {self.data_dir}")
        
        # Create class mapping
        class_names = sorted([d.name for d in class_dirs])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Load images
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            # Collect all image files (glob returns iterators, so convert to list)
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            for img_file in image_files:
                self.images.append(str(img_file))
                self.labels.append(class_idx)
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        print(f"Loaded {len(self.images)} images from {len(class_names)} classes", flush=True)
        print(f"Classes: {class_names}", flush=True)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}", flush=True)
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    @property
    def classes(self):
        return list(self.class_to_idx.keys())

def get_data_transforms(img_size=224):
    """Get data transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(model_name='resnet18', num_classes=5, pretrained=True):
    """Create ResNet model"""
    model_map = {
        'resnet18': (resnet18, ResNet18_Weights.IMAGENET1K_V1),
        'resnet34': (resnet34, ResNet34_Weights.IMAGENET1K_V1),
        'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V1)
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(model_map.keys())}")
    
    model_fn, weights = model_map[model_name]
    
    if pretrained:
        model = model_fn(weights=weights)
    else:
        model = model_fn(weights=None)
    
    # Modify final layer for number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Progress
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}', flush=True)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def main():
    # Get default epochs from environment variable
    default_epochs = int(os.getenv('DEFAULT_EPOCHS', '10'))
    
    parser = argparse.ArgumentParser(description='ResNet18 Classification Training')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset with train/valid/test folders')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save model and results')
    parser.add_argument('--epochs', type=int, default=default_epochs, help=f'Number of epochs (default: {default_epochs})')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--model', type=str, default='resnet18', help='Model: resnet18, resnet34, resnet50')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Validate paths
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}", flush=True)
        sys.exit(1)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle different dataset structures
    # Check for 100.v1i.folder structure first
    roboflow_dir = dataset_path / '100.v1i.folder'
    if roboflow_dir.exists() and roboflow_dir.is_dir():
        print(f"Found 100.v1i.folder structure, using it...", flush=True)
        dataset_path = roboflow_dir
    
    # Check for classification structure
    classification_dir = dataset_path / 'classification'
    if classification_dir.exists() and classification_dir.is_dir():
        print(f"Found classification structure, using it...", flush=True)
        dataset_path = classification_dir
    
    # Check for train and valid folders
    train_dir = dataset_path / 'train'
    valid_dir = dataset_path / 'valid'
    if not valid_dir.exists():
        valid_dir = dataset_path / 'val'
    
    if not train_dir.exists():
        print(f"ERROR: train/ folder not found in {dataset_path}", flush=True)
        print(f"Checked paths:", flush=True)
        print(f"  - {dataset_path / 'train'}", flush=True)
        print(f"  - {dataset_path / '100.v1i.folder' / 'train'}", flush=True)
        print(f"  - {dataset_path / 'classification' / 'train'}", flush=True)
        sys.exit(1)
    
    if not valid_dir.exists():
        print(f"ERROR: valid/ or val/ folder not found in {dataset_path}", flush=True)
        print(f"Checked paths:", flush=True)
        print(f"  - {dataset_path / 'valid'}", flush=True)
        print(f"  - {dataset_path / 'val'}", flush=True)
        print(f"  - {dataset_path / '100.v1i.folder' / 'valid'}", flush=True)
        print(f"  - {dataset_path / 'classification' / 'val'}", flush=True)
        sys.exit(1)
    
    print(f"Dataset path: {dataset_path}", flush=True)
    print(f"Save directory: {save_dir}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Image size: {args.img_size}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Number of classes: {args.num_classes}", flush=True)
    print("=" * 60, flush=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(args.img_size)
    
    # Load datasets
    print("\nLoading datasets...", flush=True)
    print(f"Dataset base path: {dataset_path}", flush=True)
    print(f"Train directory: {train_dir}", flush=True)
    print(f"Valid directory: {valid_dir}", flush=True)
    
    # For train, use dataset_path directly (it already points to the right location)
    train_dataset = ClassificationDataset(dataset_path, transform=train_transform, split='train')
    
    # For validation, use the parent directory and the split name
    # If valid_dir is dataset_path/valid, then split='valid'
    # If valid_dir is dataset_path/val, then split='val'
    val_split_name = valid_dir.name  # 'valid' or 'val'
    val_dataset = ClassificationDataset(dataset_path, transform=val_transform, split=val_split_name)
    
    # Verify number of classes matches
    actual_num_classes = len(train_dataset.classes)
    if actual_num_classes != args.num_classes:
        print(f"WARNING: Dataset has {actual_num_classes} classes, but num_classes={args.num_classes} was specified", flush=True)
        print(f"Using actual number of classes: {actual_num_classes}", flush=True)
        args.num_classes = actual_num_classes
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\nCreating {args.model} model with {args.num_classes} classes...", flush=True)
    model = create_model(args.model, args.num_classes, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = save_dir / 'best_model.pth'
    
    # Training loop
    print("\n" + "=" * 60, flush=True)
    print("TRAINING STARTED", flush=True)
    print("=" * 60, flush=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}", flush=True)
        print("-" * 60, flush=True)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch + 1} Summary:", flush=True)
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%", flush=True)
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%", flush=True)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': args.num_classes,
                'classes': train_dataset.classes,
                'model_name': args.model
            }, best_model_path)
            print(f"  [OK] New best model saved! (Val Acc: {val_acc:.2f}%)", flush=True)
    
    # Save final model
    final_model_path = save_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'num_classes': args.num_classes,
        'classes': train_dataset.classes,
        'model_name': args.model
    }, final_model_path)
    
    # Save training history
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save training info
    info = {
        'dataset_path': str(dataset_path),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'model': args.model,
        'num_classes': args.num_classes,
        'learning_rate': args.learning_rate,
        'classes': train_dataset.classes,
        'best_val_acc': best_val_acc,
        'final_val_acc': val_acc,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'completed_at': datetime.now().isoformat()
    }
    
    info_path = save_dir / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60, flush=True)
    print("TRAINING COMPLETED", flush=True)
    print("=" * 60, flush=True)
    print(f"Best validation accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"Final validation accuracy: {val_acc:.2f}%", flush=True)
    print(f"Model saved to: {best_model_path}", flush=True)
    print(f"Results saved to: {save_dir}", flush=True)

if __name__ == '__main__':
    main()

