"""
Simple example demonstrating pyable_dataloader functionality

This example creates synthetic medical images and shows how to use
the PyableDataset with transforms and PyTorch DataLoader.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import tempfile
import json
from pathlib import Path
import SimpleITK as sitk

from pyable_dataloader import PyableDataset, Compose, IntensityNormalization, RandomFlip, RandomTranslation


def create_synthetic_image(size=(64, 64, 64), spacing=(1.0, 1.0, 1.0)):
    """Create synthetic 3D medical image."""
    # Create a gradient pattern with some noise
    x, y, z = np.meshgrid(np.linspace(-1, 1, size[0]),
                         np.linspace(-1, 1, size[1]),
                         np.linspace(-1, 1, size[2]), indexing='ij')

    # Create ellipsoid pattern
    pattern = np.exp(-(x**2 + y**2 + 2*z**2))
    noise = np.random.randn(*size) * 0.1
    arr = (pattern + noise) * 1000 + 500

    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing(spacing)
    img.SetOrigin([0.0, 0.0, 0.0])
    return img


def create_synthetic_roi(size=(64, 64, 64), spacing=(1.0, 1.0, 1.0)):
    """Create synthetic ROI with labels."""
    arr = np.zeros(size, dtype=np.uint8)

    # Create a sphere in the center
    center = np.array(size) // 2
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < 15:
                    arr[i, j, k] = 1

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin([0.0, 0.0, 0.0])
    return img


def main():
    """Demonstrate pyable_dataloader usage."""

    print("Creating synthetic dataset...")

    # Create temporary directory for synthetic data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic subjects
        subjects = {}
        for i in range(5):
            subj_id = f"subject_{i:02d}"

            # Create T1-weighted image
            t1_img = create_synthetic_image()
            t1_path = tmpdir / f"{subj_id}_T1.nii.gz"
            sitk.WriteImage(t1_img, str(t1_path))

            # Create T2-weighted image
            t2_img = create_synthetic_image()
            t2_path = tmpdir / f"{subj_id}_T2.nii.gz"
            sitk.WriteImage(t2_img, str(t2_path))

            # Create ROI
            roi_img = create_synthetic_roi()
            roi_path = tmpdir / f"{subj_id}_roi.nii.gz"
            sitk.WriteImage(roi_img, str(roi_path))

            subjects[subj_id] = {
                'images': [str(t1_path), str(t2_path)],
                'rois': [str(roi_path)],
                'labelmaps': [],
                'label': float(i % 2)  # Binary classification
            }

        # Create manifest
        manifest_path = tmpdir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(subjects, f)

        print(f"Created manifest with {len(subjects)} subjects")

        # Create training transforms
        train_transforms = Compose([
            IntensityNormalization(method='zscore'),
            RandomFlip(axes=[0, 1, 2], prob=0.5),
            RandomTranslation(translation_range=[(-5, 5), (-5, 5), (-5, 5)], prob=0.5)
        ])

        # Create dataset
        dataset = PyableDataset(
            manifest=str(manifest_path),
            target_size=[32, 32, 32],
            target_spacing=2.0,
            mask_with_roi=True,
            transforms=train_transforms,
            stack_channels=True,
            cache_dir=str(tmpdir / 'cache'),
            return_meta=False
        )

        print(f"Dataset created with {len(dataset)} samples")

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0  # Use 0 for debugging
        )

        print("Testing DataLoader...")

        # Test a few batches
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {batch['images'].shape}")
            print(f"  Labels: {batch['label']}")
            print(f"  Image range: {batch['images'].min():.3f} to {batch['images'].max():.3f}")

            if batch_idx >= 2:  # Test 3 batches
                break

        print("\nExample completed successfully!")
        print("The pyable_dataloader is working correctly with transforms and DataLoader integration.")


if __name__ == "__main__":
    main()