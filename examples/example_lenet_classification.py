"""
Example 4: Classification Training with LeNet

This example demonstrates how to use pyable-dataloader for training a LeNet
classifier on medical images. The manifest includes classification labels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import tempfile

from pyable_dataloader import PyableDataset


class LeNet3D(nn.Module):
    """3D LeNet for classification of medical images."""

    def __init__(self, num_classes=2):
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=5, stride=1, padding=0),  # 32x32x32 -> 28x28x28
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 28x28x28 -> 14x14x14

            nn.Conv3d(6, 16, kernel_size=5, stride=1, padding=0),  # 14x14x14 -> 10x10x10
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 10x10x10 -> 5x5x5
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def create_sample_classification_data():
    """Create sample synthetic medical images with classification labels."""
    import SimpleITK as sitk

    tmpdir = Path(tempfile.mkdtemp())
    subjects = {}

    for i in range(20):  # 20 subjects for classification
        subj_id = f"sub{i:03d}"

        # Create synthetic image (32x32x32) with class-specific patterns
        if i < 10:
            # Class 0: More uniform intensity
            img_arr = np.random.randn(32, 32, 32).astype(np.float32) * 50 + 400
            label = 0
        else:
            # Class 1: More variable intensity with patterns
            img_arr = np.random.randn(32, 32, 32).astype(np.float32) * 100 + 500
            # Add some structured patterns
            img_arr[10:20, 10:20, 10:20] += np.random.randn(10, 10, 10) * 200
            label = 1

        img = sitk.GetImageFromArray(img_arr)
        img.SetSpacing([1.0, 1.0, 1.0])
        img_path = tmpdir / f"{subj_id}_T1.nii.gz"
        sitk.WriteImage(img, str(img_path))

        subjects[subj_id] = {
            'images': [str(img_path)],
            'rois': [],  # No ROIs for classification
            'labelmaps': [],
            'label': label  # Classification label
        }

    # Create manifest
    manifest_path = tmpdir / 'classification_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(subjects, f)

    return manifest_path, tmpdir


def train_lenet_classifier():
    """Train the LeNet classifier."""
    print("Creating sample classification data...")
    manifest_path, tmpdir = create_sample_classification_data()

    print("Setting up dataset and dataloader...")
    dataset = PyableDataset(
        manifest=str(manifest_path),
        target_size=[32, 32, 32],
        target_spacing=1.0,
        return_meta=False
    )

    # Split into train/val (simple split for demo)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print("Setting up LeNet model...")
    model = LeNet3D(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Training LeNet on {device}...")

    # Training loop
    num_epochs = 10
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            images = batch['images'].to(device)  # Shape: [B, C, D, H, W]
            labels = batch['label'].to(device).long()    # Shape: [B] - convert to long for CrossEntropy

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                labels = batch['label'].to(device).long()

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(".4f"
              ".4f")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'lenet_best.pth')

    print("Training completed!")
    print(".2f")
    print(f"Sample data saved in: {tmpdir}")

    # Save final model
    torch.save(model.state_dict(), 'lenet_final.pth')
    print("Models saved as 'lenet_best.pth' and 'lenet_final.pth'")


def evaluate_model():
    """Evaluate the trained model on test data."""
    print("\nEvaluating model...")

    # Load the best model
    model = LeNet3D(num_classes=2)
    model.load_state_dict(torch.load('lenet_best.pth'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create test data
    manifest_path, tmpdir = create_sample_classification_data()

    dataset = PyableDataset(
        manifest=str(manifest_path),
        target_size=[32, 32, 32],
        target_spacing=1.0,
        return_meta=False
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Evaluate
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            labels = batch['label'].to(device).long()

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    overall_acc = 100. * correct / total
    class0_acc = 100. * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    class1_acc = 100. * class_correct[1] / class_total[1] if class_total[1] > 0 else 0

    print(".2f")
    print(".2f")
    print(".2f")


if __name__ == '__main__':
    train_lenet_classifier()
    evaluate_model()