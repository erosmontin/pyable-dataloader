"""
Example usage similar to the original UnifiedNiftiDataset

This example shows how to replicate the functionality of your original
dataloader using the new pyable-based implementation.
"""

import torch
from torch.utils.data import DataLoader
from pyable_dataloader import PyableDataset, Compose, IntensityNormalization, RandomFlip, RandomAffine, RandomNoise


def create_training_dataset(
    csv_file='train2.csv',
    csv_roi_file='trainROI2.csv',
    target_size=[50, 50, 50],
    target_spacing=2.0,
    femur_z_target=None,
    cache_dir='tmp'
):
    """
    Create training dataset with augmentation.
    
    This replicates the original UnifiedNiftiDataset behavior:
    - Loads images from CSV
    - Applies ROI masking
    - Centers images in volume (optionally at femur target)
    - Applies augmentation
    - Caches results
    """
    
    # Create augmentation pipeline (similar to original _apply_augmentation)
    train_transforms = Compose([
        IntensityNormalization(method='zscore'),
        RandomFlip(axes=[0, 1, 2], prob=0.5),
        RandomAffine(
            rotation_range=5.0,
            zoom_range=(0.99, 1.01),
            shift_range=2.0,
            prob=0.5
        ),
        RandomNoise(std=0.01, prob=0.5)
    ])
    
    # ROI center target (if femur_z_target specified)
    roi_center = None
    if femur_z_target is not None:
        # Center at specified Z coordinate (X, Y will be auto-centered)
        roi_center = [0, 0, femur_z_target]
    
    # Create dataset
    # Note: You need to merge your CSV and ROI CSV into a single manifest
    # or convert them to the new format. See convert_csv_to_manifest() below
    dataset = PyableDataset(
        manifest=csv_file,  # Will be converted from CSV
        target_size=target_size,
        target_spacing=target_spacing,
        mask_with_roi=True,
        roi_labels=[1, 34, 35, 2, 36, 37],  # Femur and acetabulum labels
        roi_center_target=roi_center,
        transforms=train_transforms,
        stack_channels=True,
        cache_dir=cache_dir,
        return_meta=False,
        orientation='LPS'
    )
    
    return dataset


def create_test_dataset(
    csv_file='test2.csv',
    csv_roi_file='testROI2.csv',
    target_size=[50, 50, 50],
    target_spacing=2.0,
    cache_dir='tmp'
):
    """
    Create test dataset without augmentation.
    
    Returns transform info for overlaying results.
    """
    dataset = PyableDataset(
        manifest=csv_file,
        target_size=target_size,
        target_spacing=target_spacing,
        mask_with_roi=True,
        roi_labels=[1, 34, 35, 2, 36, 37],
        transforms=None,  # No augmentation for test
        stack_channels=True,
        cache_dir=cache_dir,
        return_meta=True,  # Return metadata for overlay
        orientation='LPS'
    )
    
    return dataset


def convert_csv_to_manifest(csv_file, csv_roi_file=None, output_json='manifest.json'):
    """
    Convert your original CSV format to the new manifest format.
    
    Original format:
        CSV: label, image_path_1, image_path_2, ...
        ROI CSV: roi_path_1, roi_path_2, ...
    
    New format:
        JSON: {
            "0": {
                "images": [image_path_1, ...],
                "rois": [roi_path_1, ...],
                "label": label_value
            }
        }
    """
    import pandas as pd
    import json
    
    # Load main CSV
    df = pd.read_csv(csv_file)
    
    # Load ROI CSV if provided
    roi_df = None
    if csv_roi_file:
        roi_df = pd.read_csv(csv_roi_file, header=None)
    
    manifest = {}
    
    for idx, row in df.iterrows():
        subject_id = str(idx)
        
        # First column is label
        label = float(row.iloc[0])
        
        # Rest are image paths
        images = []
        for val in row.iloc[1:]:
            if pd.notna(val) and len(str(val)) > 3:
                images.append(str(val))
        
        # Get ROI path
        rois = []
        if roi_df is not None and idx < len(roi_df):
            roi_path = str(roi_df.iloc[idx, 0])
            if len(roi_path) > 3:
                rois.append(roi_path)
        
        manifest[subject_id] = {
            'images': images,
            'rois': rois,
            'labelmaps': [],
            'label': label
        }
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Converted {len(manifest)} subjects to {output_json}")
    return output_json


def overlay_prediction_on_original(dataset, subject_idx, prediction_array):
    """
    Overlay model prediction back to original image space.
    
    Args:
        dataset: PyableDataset instance (with return_meta=True)
        subject_idx: Index of subject
        prediction_array: Model output as numpy array (Z, Y, X)
    
    Returns:
        SimpleITK image in original space
    """
    subject_id = dataset.ids[subject_idx]
    overlayer = dataset.get_original_space_overlayer(subject_id)
    
    # Overlay to original space
    original_space_sitk = overlayer(prediction_array, interpolator='linear')
    
    return original_space_sitk


if __name__ == "__main__":
    # Convert CSVs to manifest format (do this once)
    print("Converting CSV to manifest format...")
    manifest_train = convert_csv_to_manifest('train2.csv', 'trainROI2.csv', 'train_manifest.json')
    manifest_test = convert_csv_to_manifest('test2.csv', 'testROI2.csv', 'test_manifest.json')
    
    # Create training dataset
    print("\nCreating training dataset...")
    train_dataset = create_training_dataset(
        csv_file=manifest_train,
        target_size=[50, 50, 50],
        target_spacing=2.0,
        cache_dir='cache/train'
    )
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = create_test_dataset(
        csv_file=manifest_test,
        target_size=[50, 50, 50],
        target_spacing=2.0,
        cache_dir='cache/test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    # Test training loop
    print("\nTesting training loop...")
    for batch_idx, batch in enumerate(train_loader):
        images = batch['images']  # B × C × D × H × W
        labels = batch['label']    # B
        
        print(f"Batch {batch_idx}: images shape = {images.shape}, labels = {labels}")
        
        if batch_idx >= 2:
            break
    
    # Test inference with overlay
    print("\nTesting inference with overlay...")
    sample = test_dataset[0]
    images = sample['images']
    label = sample['label']
    meta = sample['meta']
    
    print(f"Test sample: images shape = {images.shape}, label = {label}")
    print(f"Spacing: {meta['spacing']}")
    print(f"Origin: {meta['origin']}")
    print(f"Size: {meta['size']}")
    
    # Simulate model prediction
    fake_prediction = torch.sigmoid(torch.randn_like(images[0])).numpy()
    
    # Overlay back to original space
    print("\nOverlaying prediction to original space...")
    original_space_sitk = overlay_prediction_on_original(test_dataset, 0, fake_prediction)
    
    print(f"Original space size: {original_space_sitk.GetSize()}")
    print(f"Original space spacing: {original_space_sitk.GetSpacing()}")
    
    # Save result
    import SimpleITK as sitk
    sitk.WriteImage(original_space_sitk, 'test_prediction_original_space.nii.gz')
    print("Saved prediction to test_prediction_original_space.nii.gz")
    
    print("\n✅ All tests passed!")
