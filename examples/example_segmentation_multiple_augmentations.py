"""
Example 3: Segmentation Training with Multiple Augmentations

This example demonstrates how to use the get_multiple_augmentations() method
to generate multiple augmented versions of each sample and stack them into batches
for training a segmentation model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import json
import tempfile

from pyable_dataloader import PyableDataset, Compose, RandomRotation, RandomTranslation, IntensityNormalization


class SimpleSegmentationModel(nn.Module):
    """Simple 3D CNN for segmentation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, out_channels, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AugmentedDataset(Dataset):
    """Dataset that generates multiple augmentations per sample."""

    def __init__(self, base_dataset, augmentation_configs, base_seed=42):
        self.base_dataset = base_dataset
        self.augmentation_configs = augmentation_configs
        self.base_seed = base_seed

        # Pre-generate all augmented samples for efficiency
        self.augmented_data = []
        for subj_idx in range(len(base_dataset)):
            augmented_samples = base_dataset.get_multiple_augmentations(
                subject_idx=subj_idx,
                augmentation_configs=augmentation_configs,
                base_seed=base_seed
            )
            self.augmented_data.extend(augmented_samples)

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        sample = self.augmented_data[idx]

        # Convert to tensors
        images = torch.from_numpy(sample['images'][0]).unsqueeze(0).float()  # Add channel dim
        roi = torch.from_numpy(sample['rois'][0]).float()

        return {
            'images': images,
            'rois': [roi],  # Keep as list for compatibility
            'augmentation_name': sample['name'],
            'subject_id': sample['meta']['subject_id']
        }


def create_sample_data():
    """Create sample synthetic medical images and ROIs for demonstration."""
    import SimpleITK as sitk

    tmpdir = Path(tempfile.mkdtemp())
    subjects = {}

    for i in range(5):  # 5 subjects (fewer since we'll augment each)
        subj_id = f"sub{i:03d}"

        # Create synthetic image (32x32x32)
        img_arr = np.random.randn(32, 32, 32).astype(np.float32) * 100 + 500
        img = sitk.GetImageFromArray(img_arr)
        img.SetSpacing([1.0, 1.0, 1.0])
        img_path = tmpdir / f"{subj_id}_T1.nii.gz"
        sitk.WriteImage(img, str(img_path))

        # Create synthetic ROI (spherical region)
        roi_arr = np.zeros((32, 32, 32), dtype=np.uint8)
        center = np.array([16, 16, 16])
        for x in range(32):
            for y in range(32):
                for z in range(32):
                    if np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2) < 8:
                        roi_arr[x, y, z] = 1

        roi = sitk.GetImageFromArray(roi_arr)
        roi.SetSpacing([1.0, 1.0, 1.0])
        roi_path = tmpdir / f"{subj_id}_roi.nii.gz"
        sitk.WriteImage(roi, str(roi_path))

        subjects[subj_id] = {
            'images': [str(img_path)],
            'rois': [str(roi_path)],
            'labelmaps': []
        }

    # Create manifest
    manifest_path = tmpdir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(subjects, f)

    return manifest_path, tmpdir


def dice_loss(pred, target, smooth=1e-5):
    """Dice loss for segmentation."""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def train_segmentation_with_multiple_augmentations():
    """Train the segmentation model using multiple augmentations per sample."""
    print("Creating sample data...")
    manifest_path, tmpdir = create_sample_data()

    # Define multiple augmentation strategies
    augmentation_configs = [
        {
            'name': 'original',
            'transforms': Compose([IntensityNormalization(method='zscore')]),
        },
        {
            'name': 'light_rotation',
            'transforms': Compose([
                IntensityNormalization(method='zscore'),
                RandomRotation(rotation_range=[[-5, 5], [-5, 5], [-5, 5]], prob=1.0)
            ])
        },
        {
            'name': 'moderate_augmentation',
            'transforms': Compose([
                IntensityNormalization(method='zscore'),
                RandomRotation(rotation_range=[[-10, 10], [-10, 10], [-5, 5]], prob=1.0),
                RandomTranslation(translation_range=[[-3, 3], [-3, 3], [-2, 2]], prob=1.0)
            ])
        }
    ]

    print("Setting up base dataset...")
    base_dataset = PyableDataset(
        manifest=str(manifest_path),
        target_size=[32, 32, 32],
        target_spacing=1.0,
        return_meta=True
    )

    print("Creating augmented dataset...")
    augmented_dataset = AugmentedDataset(
        base_dataset=base_dataset,
        augmentation_configs=augmentation_configs,
        base_seed=42
    )

    print(f"Original dataset: {len(base_dataset)} subjects")
    print(f"Augmented dataset: {len(augmented_dataset)} samples")
    print(f"Augmentations per subject: {len(augmentation_configs)}")

    dataloader = DataLoader(augmented_dataset, batch_size=4, shuffle=True)

    print("Setting up model...")
    model = SimpleSegmentationModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Training with multiple augmentations on {device}...")

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        augmentation_counts = {}

        for batch in dataloader:
            images = batch['images'].to(device)  # Shape: [B, C, D, H, W]
            rois = batch['rois'][0].to(device)    # Shape: [B, D, H, W] (single ROI)
            aug_names = batch['augmentation_name']

            # Forward pass
            outputs = model(images)
            loss = dice_loss(outputs, rois.unsqueeze(1))  # Add channel dim to target

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Count augmentations in this batch
            for name in aug_names:
                augmentation_counts[name] = augmentation_counts.get(name, 0) + 1

        avg_loss = epoch_loss / len(dataloader)
        print(".4f")
        print(f"  Augmentation distribution: {augmentation_counts}")

    print("Training with multiple augmentations completed!")
    print(f"Sample data saved in: {tmpdir}")

    # Save model
    torch.save(model.state_dict(), 'segmentation_model_multi_aug.pth')
    print("Model saved as 'segmentation_model_multi_aug.pth'")


if __name__ == '__main__':
    train_segmentation_with_multiple_augmentations()