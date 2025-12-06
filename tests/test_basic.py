"""
Basic tests for pyable_dataloader

Run with: pytest tests/test_basic.py
"""

import pytest
import numpy as np
import torch
import SimpleITK as sitk
import tempfile
import json
from pathlib import Path

from pyable_dataloader import PyableDataset, IntensityNormalization, RandomFlip, Compose, RandomTranslation, RandomRotation


def create_synthetic_image(size=(32, 32, 32), spacing=(1.0, 1.0, 1.0)):
    """Create synthetic 3D medical image."""
    arr = np.random.randn(*size).astype(np.float32) * 100 + 500
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin([0.0, 0.0, 0.0])
    return img


def create_synthetic_roi(size=(32, 32, 32), spacing=(1.0, 1.0, 1.0)):
    """Create synthetic ROI with labels."""
    arr = np.zeros(size, dtype=np.uint8)
    # Create a sphere in the center
    center = np.array(size) // 2
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < 8:
                    arr[i, j, k] = 1
    
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin([0.0, 0.0, 0.0])
    return img


@pytest.fixture
def temp_dataset():
    """Create temporary dataset with synthetic images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic images
        subjects = {}
        for i in range(3):
            subj_id = f"sub{i:03d}"
            
            # Create image
            img = create_synthetic_image()
            img_path = tmpdir / f"{subj_id}_T1.nii.gz"
            sitk.WriteImage(img, str(img_path))
            
            # Create ROI
            roi = create_synthetic_roi()
            roi_path = tmpdir / f"{subj_id}_roi.nii.gz"
            sitk.WriteImage(roi, str(roi_path))
            
            subjects[subj_id] = {
                'images': [str(img_path)],
                'rois': [str(roi_path)],
                'labelmaps': [],
                'label': float(i % 2)
            }
        
        # Create manifest
        manifest_path = tmpdir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(subjects, f)
        
        yield manifest_path, tmpdir


def test_dataset_creation(temp_dataset):
    """Test basic dataset creation."""
    manifest_path, tmpdir = temp_dataset
    
    dataset = PyableDataset(
        manifest=str(manifest_path),
        target_size=[16, 16, 16],
        target_spacing=2.0,
        return_meta=True
    )
    
    assert len(dataset) == 3
    
    # Get first sample
    sample = dataset[0]
    
    assert 'images' in sample
    assert 'rois' in sample
    assert 'label' in sample
    assert 'meta' in sample
    
    # Check tensor shapes
    images = sample['images']
    assert isinstance(images, torch.Tensor)
    assert images.shape[-3:] == (16, 16, 16)  # D, H, W


def test_multi_channel_stacking(temp_dataset):
    """Test stacking multiple images as channels."""
    manifest_path, tmpdir = temp_dataset
    
    # Load manifest and add second image
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Create second image for first subject
    img2 = create_synthetic_image()
    img2_path = tmpdir / 'sub000_T2.nii.gz'
    sitk.WriteImage(img2, str(img2_path))
    
    manifest['sub000']['images'].append(str(img2_path))
    
    dataset = PyableDataset(
        manifest=manifest,
        target_size=[16, 16, 16],
        target_spacing=2.0,
        stack_channels=True
    )
    
    sample = dataset[0]
    images = sample['images']
    
    # Should have 2 channels
    assert images.shape[0] == 2
    assert images.shape[1:] == (16, 16, 16)


def test_roi_masking(temp_dataset):
    """Test ROI masking functionality."""
    manifest_path, tmpdir = temp_dataset
    
    dataset_no_mask = PyableDataset(
        manifest=str(manifest_path),
        target_size=[16, 16, 16],
        mask_with_roi=False
    )
    
    dataset_with_mask = PyableDataset(
        manifest=str(manifest_path),
        target_size=[16, 16, 16],
        mask_with_roi=True
    )
    
    sample_no_mask = dataset_no_mask[0]
    sample_with_mask = dataset_with_mask[0]
    
    # Masked version should have zeros outside ROI
    images_no_mask = sample_no_mask['images'].numpy()
    images_with_mask = sample_with_mask['images'].numpy()
    
    # Should have fewer non-zero voxels when masked
    assert np.count_nonzero(images_with_mask) <= np.count_nonzero(images_no_mask)


def test_label_preservation(temp_dataset):
    """Test that label values are preserved (no interpolation)."""
    manifest_path, tmpdir = temp_dataset
    
    dataset = PyableDataset(
        manifest=str(manifest_path),
        target_size=[16, 16, 16],
        target_spacing=2.0
    )
    
    sample = dataset[0]
    roi = sample['rois'][0].numpy()
    
    # Check that only label values 0 and 1 exist (no interpolated values)
    unique_values = np.unique(roi)
    assert all(v in [0, 1] for v in unique_values), f"Found interpolated values: {unique_values}"


def test_dataset_with_pyable_transform(temp_dataset):
    """Test that dataset applies pyable-backed RandomTranslation and preserves labels"""
    pytest.importorskip('torch')

    manifest_path, tmpdir = temp_dataset
    dataset = PyableDataset(
        manifest=str(manifest_path),
        target_size=[16, 16, 16],
        target_spacing=2.0,
        transforms=Compose([RandomTranslation(translation_range=[[-3,3],[-3,3],[-3,3]], prob=1.0)])
    )

    sample = dataset[0]
    roi = sample['rois'][0].numpy()
    unique_vals = np.unique(roi)
    assert all(v in [0, 1] for v in unique_vals)


def test_intensity_normalization():
    """Test intensity normalization transform."""
    images = np.random.randn(32, 32, 32).astype(np.float32) * 100 + 500
    rois = []
    labelmaps = []
    meta = {}
    
    transform = IntensityNormalization(method='zscore')
    
    images_norm, _, _ = transform(images, rois, labelmaps, meta)
    
    # Check that mean ≈ 0 and std ≈ 1
    mean = images_norm.mean()
    std = images_norm.std()
    
    assert abs(mean) < 0.1, f"Mean not close to 0: {mean}"
    assert abs(std - 1.0) < 0.1, f"Std not close to 1: {std}"


def test_random_flip():
    """Test random flip transform."""
    np.random.seed(42)
    
    images = np.random.randn(32, 32, 32).astype(np.float32)
    rois = [np.random.randint(0, 2, (32, 32, 32)).astype(np.uint8)]
    labelmaps = []
    meta = {}
    
    transform = RandomFlip(axes=[0], prob=1.0)  # Always flip axis 0
    
    images_flipped, rois_flipped, _ = transform(images, rois, labelmaps, meta)
    
    # Check that first slice becomes last slice
    assert np.allclose(images[0], images_flipped[-1])
    assert np.allclose(rois[0][0], rois_flipped[0][-1])


def test_caching(temp_dataset):
    """Test caching functionality."""
    manifest_path, tmpdir = temp_dataset
    cache_dir = tmpdir / 'cache'
    
    dataset = PyableDataset(
        manifest=str(manifest_path),
        target_size=[16, 16, 16],
        cache_dir=str(cache_dir)
    )
    
    # First access - should create cache
    sample1 = dataset[0]
    
    # Check that cache was created
    cache_files = list(cache_dir.glob('*.npz'))
    assert len(cache_files) > 0
    
    # Second access - should load from cache
    sample2 = dataset[0]
    
    # Should be identical
    assert torch.allclose(sample1['images'], sample2['images'])


def test_dataloader_integration(temp_dataset):
    """Test integration with PyTorch DataLoader."""
    from torch.utils.data import DataLoader
    
    manifest_path, tmpdir = temp_dataset
    
    dataset = PyableDataset(
        manifest=str(manifest_path),
        target_size=[16, 16, 16]
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    batch = next(iter(loader))
    
    assert 'images' in batch
    assert 'label' in batch
    
    # Check batch dimensions
    images = batch['images']
    labels = batch['label']
    
    assert images.shape[0] == 2  # Batch size
    assert labels.shape[0] == 2


def test_overlay_function(temp_dataset):
    """Test overlay back to original space."""
    manifest_path, tmpdir = temp_dataset
    
    dataset = PyableDataset(
        manifest=str(manifest_path),
        target_size=[16, 16, 16],
        return_meta=True
    )
    
    # Get overlayer
    overlayer = dataset.get_original_space_overlayer('sub000')
    
    # Create fake prediction
    prediction = np.random.rand(16, 16, 16).astype(np.float32)
    
    # Overlay to original space
    original_space_sitk = overlayer(prediction)
    
    # Check that it's a valid SimpleITK image
    assert isinstance(original_space_sitk, sitk.Image)
    assert original_space_sitk.GetSize() == (32, 32, 32)  # Original size


def test_get_multiple_augmentations(temp_dataset):
    """Test get_multiple_augmentations method."""
    manifest_path, tmpdir = temp_dataset
    
    dataset = PyableDataset(
        manifest=str(manifest_path),
        target_size=[16, 16, 16],
        return_meta=True
    )
    
    # Define augmentation configs
    augmentation_configs = [
        {
            'name': 'rotation_only',
            'transforms': Compose([
                IntensityNormalization(method='zscore'),
                RandomRotation(rotation_range=[[-5, 5], [-5, 5], [-5, 5]], prob=1.0)
            ])
        },
        {
            'name': 'translation_only',
            'transforms': Compose([
                IntensityNormalization(method='zscore'),
                RandomTranslation(translation_range=[[-2, 2], [-2, 2], [-1, 1]], prob=1.0)
            ])
        }
    ]
    
    # Get multiple augmentations
    results = dataset.get_multiple_augmentations(
        subject_idx=0,
        augmentation_configs=augmentation_configs,
        base_seed=42
    )
    
    # Check results
    assert len(results) == 2
    assert results[0]['name'] == 'rotation_only'
    assert results[1]['name'] == 'translation_only'
    
    # Check that each has the required keys
    for result in results:
        assert 'images' in result
        assert 'rois' in result
        assert 'labelmaps' in result
        assert 'meta' in result
        assert 'config' in result
        assert result['config']['name'] == result['name']
    
    # Check reproducibility: same seed should give same results
    results2 = dataset.get_multiple_augmentations(
        subject_idx=0,
        augmentation_configs=augmentation_configs,
        base_seed=42
    )
    
    # Compare images (should be identical due to same seed)
    np.testing.assert_array_equal(results[0]['images'], results2[0]['images'])
    np.testing.assert_array_equal(results[1]['images'], results2[1]['images'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
