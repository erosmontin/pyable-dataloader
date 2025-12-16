"""
Example: Angle Prediction Dataset from patient.json

This example demonstrates how to create a custom dataset that:
1. Uses 'original_image' as the input image
2. Uses 'original_roi' as the labelmap/segmentation
3. Returns 'angle' as the regression target for your network
4. Applies data augmentation transforms
5. Saves debug images for visualization

Your patient.json should have a structure like:
{
    "patient_001": {
        "original_image": "/path/to/image.nii.gz",
        "original_roi": "/path/to/roi.nii.gz",
        "angle": 45.5,
        ...
    },
    ...
}
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Union, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk

from pyable_dataloader import PyableDataset

try:
    from pyable_dataloader import (
        Compose,
        IntensityNormalization,
        RandomTranslation,
        RandomRotation,
        RandomBSpline,
        RandomFlip,
        RandomNoise
    )
except ImportError:
    print("Warning: pyable_dataloader transforms not found. Using basic transforms.")
    Compose = None


def convert_patient_json_to_manifest(
    patient_json_path: str,
    image_key: str = "original_image",
    roi_key: str = "original_roi",
    angle_key: str = "angle",
    id_key: str = "id"
) -> dict:
    """
    Convert patient.json format to PyableDataset manifest format.
    
    Supports two formats:
    
    1. Dictionary format:
        {
            "patient_001": {
                "original_image": "/path/to/image.nii.gz",
                "original_roi": "/path/to/roi.nii.gz",
                "angle": 45.5
            }
        }
    
    2. List format:
        [
            {
                "id": "patient_001",
                "original_image": "/path/to/image.nii.gz",
                "original_roi": "/path/to/roi.nii.gz",
                "angle": 45.5
            },
            ...
        ]
    
    Converts to:
        {
            "patient_001": {
                "images": ["/path/to/image.nii.gz"],
                "labelmaps": ["/path/to/roi.nii.gz"],
                "angle": 45.5
            }
        }
    """
    if isinstance(patient_json_path, (dict, list)):
        data = patient_json_path
    else:
        with open(patient_json_path, 'r') as f:
            data = json.load(f)
    
    manifest = {}
    
    # Handle list format: [{"id": "patient_001", ...}, ...]
    if isinstance(data, list):
        for idx, subj_data in enumerate(data):
            # Get subject ID from id_key or use index
            subj_id = str(subj_data.get(id_key, f"subject_{idx:04d}"))
            
            manifest[subj_id] = {
                'images': [subj_data[image_key]] if image_key in subj_data else [],
                'labelmaps': [subj_data[roi_key]] if roi_key in subj_data and subj_data.get(roi_key) else [],
                'rois': [],
                angle_key: subj_data.get(angle_key, 0.0)
            }
            # Preserve any other metadata
            for key, value in subj_data.items():
                if key not in [id_key, image_key, roi_key, angle_key]:
                    manifest[subj_id][key] = value
    
    # Handle dict format: {"patient_001": {...}, ...}
    else:
        for subj_id, subj_data in data.items():
            manifest[subj_id] = {
                'images': [subj_data[image_key]] if image_key in subj_data else [],
                'labelmaps': [subj_data[roi_key]] if roi_key in subj_data and subj_data.get(roi_key) else [],
                'rois': [],
                angle_key: subj_data.get(angle_key, 0.0)
            }
            # Preserve any other metadata
            for key, value in subj_data.items():
                if key not in [image_key, roi_key, angle_key]:
                    manifest[subj_id][key] = value
    
    return manifest


class AnglePredictionDataset(PyableDataset):
    """
    Dataset for angle prediction from medical images.
    
    Inherits from PyableDataset and adds:
    - Angle extraction as regression target
    - Debug image saving
    - Angle normalization
    
    Args:
        manifest_path: Path to patient.json file or dict
        target_size: Target voxel dimensions [D, H, W] (e.g., [64, 64, 64])
        target_spacing: Target spacing in mm (float or [x,y,z])
        transforms: Optional transform pipeline for data augmentation
        debug_dir: Optional directory to save debug images
        debug_every_n: Save debug images every N samples (default: 10)
        normalize_angle: If True, normalize angles to [0, 1] range
        angle_range: Tuple (min, max) for angle normalization
        dtype: PyTorch dtype for output tensors
        image_key: Key in JSON for the image path (default: "original_image")
        roi_key: Key in JSON for the ROI path (default: "original_roi")
        angle_key: Key in JSON for the angle value (default: "angle")
        **kwargs: Additional arguments passed to PyableDataset
    """
    
    def __init__(
        self,
        manifest_path: Union[str, dict],
        target_size: List[int] = [64, 64, 64],
        target_spacing: Union[float, List[float]] = 1.0,
        transforms: Optional[Callable] = None,
        debug_dir: Optional[str] = None,
        debug_every_n: int = 10,
        normalize_angle: bool = False,
        angle_range: tuple = (0, 360),
        dtype: torch.dtype = torch.float32,
        image_key: str = "original_image",
        roi_key: str = "original_roi",
        angle_key: str = "angle",
        **kwargs
    ):
        # Store angle-specific settings before calling super().__init__
        self._angle_debug_dir = debug_dir  # Use different name to avoid conflict
        self.debug_every_n = debug_every_n
        self.normalize_angle = normalize_angle
        self.angle_range = angle_range
        self.angle_key = angle_key
        self._sample_count = 0
        
        # Create debug directory if specified
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
        
        # Convert patient.json format to PyableDataset manifest format
        manifest = convert_patient_json_to_manifest(
            manifest_path, 
            image_key=image_key,
            roi_key=roi_key,
            angle_key=angle_key
        )
        
        # Initialize parent PyableDataset
        super().__init__(
            manifest=manifest,
            target_size=target_size,
            target_spacing=target_spacing,
            transforms=transforms,
            dtype=dtype,
            return_meta=True,
            **kwargs
        )
        
        print(f"AnglePredictionDataset: Loaded {len(self.ids)} subjects")
    
    def _save_angle_debug(self, subject_id: str, image_array: np.ndarray, 
                          labelmap_array: np.ndarray, angle: float, 
                          augmented: bool = False):
        """Save debug images as NIfTI files with angle metadata."""
        if self._angle_debug_dir is None:
            return
        
        suffix = "_augmented" if augmented else "_original"
        
        # Handle multi-channel images (take first channel)
        if image_array.ndim == 4:
            img_to_save = image_array[0]
        else:
            img_to_save = image_array
        
        # Save image
        img_sitk = sitk.GetImageFromArray(img_to_save.astype(np.float32))
        img_sitk.SetSpacing(self.target_spacing)
        img_path = os.path.join(self._angle_debug_dir, f"{subject_id}{suffix}_image.nii.gz")
        sitk.WriteImage(img_sitk, img_path)
        
        # Save labelmap
        if labelmap_array.ndim == 4:
            lm_to_save = labelmap_array[0]
        else:
            lm_to_save = labelmap_array
        
        lm_sitk = sitk.GetImageFromArray(lm_to_save.astype(np.uint8))
        lm_sitk.SetSpacing(self.target_spacing)
        lm_path = os.path.join(self._angle_debug_dir, f"{subject_id}{suffix}_labelmap.nii.gz")
        sitk.WriteImage(lm_sitk, lm_path)
        
        # Save metadata with angle
        meta_path = os.path.join(self._angle_debug_dir, f"{subject_id}{suffix}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump({
                'subject_id': subject_id,
                'angle': angle,
                'image_shape': list(image_array.shape),
                'labelmap_shape': list(labelmap_array.shape),
                'spacing': self.target_spacing,
                'augmented': augmented
            }, f, indent=2)
        
        print(f"  Debug saved: {subject_id}{suffix}")
    
    def __getitem__(self, idx: int) -> dict:
        """Get preprocessed item with angle target."""
        # Get base result from PyableDataset
        result = super().__getitem__(idx)
        
        subject_id = result['id']
        item = self.data[subject_id]
        
        # Extract angle from manifest
        angle = float(item.get(self.angle_key, 0.0))
        
        # Additional debug save with angle metadata
        self._sample_count += 1
        should_debug = (self._angle_debug_dir is not None and 
                       self._sample_count % self.debug_every_n == 0)
        
        if should_debug:
            # Get numpy arrays from tensors for debug saving
            image_array = result['images'].numpy()
            labelmap_array = result['labelmaps'][0].numpy() if result['labelmaps'] else np.zeros_like(image_array[0])
            self._save_angle_debug(subject_id, image_array, labelmap_array, angle, 
                                   augmented=(self.transforms is not None))
        
        # Normalize angle if requested
        if self.normalize_angle:
            angle_normalized = (angle - self.angle_range[0]) / (self.angle_range[1] - self.angle_range[0])
        else:
            angle_normalized = angle
        
        # Restructure output for angle prediction task
        # Use 'image' instead of 'images' for cleaner API
        output = {
            'id': subject_id,
            'image': result['images'],                    # Shape: (C, D, H, W)
            'labelmap': result['labelmaps'][0] if result['labelmaps'] else torch.zeros_like(result['images'][0:1]),
            'angle': torch.tensor(angle_normalized, dtype=self.dtype),
            'angle_original': angle,
            'meta': result.get('meta', {})
        }
        
        return output


def create_augmentation_pipeline(
    translation_range: float = 5.0,
    rotation_range: float = 15.0,
    use_bspline: bool = True,
    use_flip: bool = True,
    flip_axes: List[int] = [2],  # Default: left-right only (X axis in LPS)
    use_noise: bool = True,
    noise_std: float = 0.05,
    normalize: bool = True,
    normalize_method: str = 'zscore'
) -> Callable:
    """
    Create a data augmentation pipeline.
    
    Args:
        translation_range: Max translation in mm (0 to disable)
        rotation_range: Max rotation in degrees (0 to disable)
        use_bspline: Whether to apply BSpline deformation
        use_flip: Whether to apply random flips
        flip_axes: Which axes to flip. In LPS orientation:
                   - 0 = Z (Superior-Inferior)
                   - 1 = Y (Anterior-Posterior) 
                   - 2 = X (Left-Right) - DEFAULT
        use_noise: Whether to apply random noise
        noise_std: Standard deviation for noise
        normalize: Whether to apply intensity normalization
        normalize_method: 'zscore', 'minmax', or 'percentile'
    
    Returns:
        Compose transform pipeline wrapped to handle list inputs
    """
    if Compose is None:
        print("Warning: Transforms not available, returning identity transform")
        return lambda images, rois, labelmaps, meta: (images, rois, labelmaps)
    
    transforms_list = []
    
    # Normalization (optional, at the start)
    if normalize:
        transforms_list.append(
            IntensityNormalization(method=normalize_method, per_channel=True)
        )
    
    if translation_range > 0:
        # translation_range is per-axis list of (min, max) in mm
        t = translation_range
        transforms_list.append(
            RandomTranslation(translation_range=[(-t, t), (-t, t), (-t, t)], prob=0.5)
        )
    
    if rotation_range > 0:
        # rotation_range is per-axis list of (min, max) in degrees
        r = rotation_range
        transforms_list.append(
            RandomRotation(rotation_range=[(-r, r), (-r, r), (-r, r)], prob=0.5)
        )
    
    if use_bspline:
        # mesh_size: control point grid, magnitude: max deformation in mm
        transforms_list.append(
            RandomBSpline(mesh_size=(4, 4, 4), magnitude=5.0, prob=0.3)
        )
    
    if use_flip:
        transforms_list.append(
            RandomFlip(axes=flip_axes, prob=0.5)
        )
    
    if use_noise:
        transforms_list.append(
            RandomNoise(std=noise_std, prob=0.5)
        )
    
    # If no transforms, return identity
    if not transforms_list:
        return lambda images, rois, labelmaps, meta: (images, rois, labelmaps)
    
    composed = Compose(transforms_list)
    
    # Wrap to handle list inputs from PyableDataset
    return ListToArrayTransformWrapper(composed)


class ListToArrayTransformWrapper:
    """
    Wrapper that handles the conversion between list and array formats.
    
    PyableDataset passes lists of arrays to transforms, but some transforms
    expect stacked numpy arrays. This wrapper handles the conversion.
    """
    
    def __init__(self, transform: Callable):
        self.transform = transform
    
    def __call__(self, images, rois, labelmaps, meta):
        # Handle empty input
        if not images or (isinstance(images, list) and len(images) == 0):
            return images, rois, labelmaps
        
        # Convert Imaginable objects to numpy if needed
        def to_numpy(obj):
            if hasattr(obj, 'getImageAsNumpy'):
                return obj.getImageAsNumpy()
            elif isinstance(obj, np.ndarray):
                return obj
            else:
                return obj  # Unknown type, pass through
        
        # Convert lists to stacked arrays if needed
        was_list = isinstance(images, list)
        num_images = len(images) if was_list else 1
        
        if was_list:
            # Convert each image to numpy
            images_np = [to_numpy(img) for img in images]
            # Stack images: list of (Z,Y,X) -> (C, Z, Y, X)
            if len(images_np) > 1:
                images_array = np.stack(images_np, axis=0)
            else:
                # Single image: add channel dim (Z,Y,X) -> (1, Z, Y, X)
                images_array = np.expand_dims(images_np[0], axis=0)
        else:
            images_array = to_numpy(images)
        
        # Convert rois and labelmaps to numpy too
        rois_np = [to_numpy(r) for r in (rois if isinstance(rois, list) else [rois])]
        labelmaps_np = [to_numpy(lm) for lm in (labelmaps if isinstance(labelmaps, list) else [labelmaps])]
        
        # Apply transform
        images_out, rois_out, labelmaps_out = self.transform(images_array, rois_np, labelmaps_np, meta)
        
        # Convert back to list if input was list
        if was_list:
            if images_out.ndim == 4:
                # (C, Z, Y, X) -> list of (Z, Y, X)
                images_out = [images_out[i] for i in range(images_out.shape[0])]
            elif images_out.ndim == 3:
                # Single image (Z, Y, X) - wrap in list
                images_out = [images_out]
        
        return images_out, rois_out, labelmaps_out


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Main example showing how to use the AnglePredictionDataset."""
    
    print("=" * 80)
    print("Angle Prediction Dataset Example")
    print("=" * 80)
    
    # Path to your patient.json
    # MODIFY THIS to point to your actual file
    manifest_path = "/data/PROJECTS/alpha_expet/patient.json"
    
    # Check if file exists
    if not os.path.exists(manifest_path):
        print(f"\n⚠️  Manifest not found at: {manifest_path}")
        print("Creating a synthetic example instead...\n")
        manifest_path = create_synthetic_manifest()
    
    debug_dir = "/tmp/angle_prediction_debug"
    
    # FIRST: Test WITHOUT augmentation to verify data loading works
    print("\n1. Creating dataset WITHOUT augmentation (to verify loading)...")
    
    dataset_no_aug = AnglePredictionDataset(
        manifest_path=manifest_path,
        target_size=[64, 64, 64],       # Resample to 64x64x64
        target_spacing=2.0,              # 2mm isotropic
        transforms=None,                 # NO augmentation
        debug_dir=debug_dir,            # Save debug images here
        debug_every_n=1,                # Save every sample
        normalize_angle=False,           # Keep angles in original range
        image_key="original_image",      # Key for image in JSON
        roi_key="original_roi",          # Key for ROI in JSON
        angle_key="angle"                # Key for angle in JSON
    )
    
    print(f"   Dataset size: {len(dataset_no_aug)} subjects")
    
    # Test loading one sample without augmentation
    print("\n2. Testing single sample load (no augmentation)...")
    sample = dataset_no_aug[0]
    print(f"   - ID: {sample['id']}")
    print(f"   - Image shape: {sample['image'].shape}")
    print(f"   - Image min/max: {sample['image'].min():.2f} / {sample['image'].max():.2f}")
    print(f"   - Labelmap shape: {sample['labelmap'].shape}")
    print(f"   - Angle: {sample['angle']}")
    
    # Check if image has content
    if sample['image'].max() == 0:
        print("\n   ⚠️  WARNING: Image is all zeros! Check the image paths in your JSON.")
    else:
        print("\n   ✓ Image has non-zero values - loading works!")
    
    # NOW: Test WITH augmentation
    print("\n3. Creating augmentation pipeline...")
    augmentation = create_augmentation_pipeline(
        translation_range=0,            # Disable translation for now
        rotation_range=0,               # Disable rotation for now  
        use_bspline=False,              # Disable BSpline
        use_flip=True,                  # Only flip
        use_noise=False,                # Disable noise
        normalize=False                 # Disable normalization
    )
    
    print("\n4. Creating dataset WITH augmentation...")
    dataset = AnglePredictionDataset(
        manifest_path=manifest_path,
        target_size=[64, 64, 64],
        target_spacing=2.0,
        transforms=augmentation,
        debug_dir=debug_dir,
        debug_every_n=1,
        normalize_angle=False,
        image_key="original_image",
        roi_key="original_roi",
        angle_key="angle"
    )
    
    # Test loading one sample with augmentation
    print("\n5. Testing single sample load (with augmentation)...")
    sample_aug = dataset[0]
    print(f"   - ID: {sample_aug['id']}")
    print(f"   - Image shape: {sample_aug['image'].shape}")
    print(f"   - Image min/max: {sample_aug['image'].min():.2f} / {sample_aug['image'].max():.2f}")
    print(f"   - Angle: {sample_aug['angle']}")
    
    if sample_aug['image'].max() == 0:
        print("\n   ⚠️  WARNING: Augmented image is all zeros!")
    else:
        print("\n   ✓ Augmented image has non-zero values!")
    
    # Create DataLoader
    print("\n6. Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    # Iterate through one epoch
    print("\n7. Iterating through dataset...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n   Batch {batch_idx + 1}:")
        print(f"   - IDs: {batch['id']}")
        print(f"   - Image shape: {batch['image'].shape}")
        print(f"   - Image min/max: {batch['image'].min():.2f} / {batch['image'].max():.2f}")
        print(f"   - Labelmap shape: {batch['labelmap'].shape}")
        print(f"   - Angles: {batch['angle']}")
        
        # In your training loop, you would do:
        # images = batch['image']           # Input to network
        # labelmaps = batch['labelmap']     # Can be used as additional input or for masking
        # angles = batch['angle']           # Target for regression
        #
        # # Forward pass
        # predicted_angles = model(images, labelmaps)  # or just model(images)
        # loss = criterion(predicted_angles, angles)
        
        if batch_idx >= 2:  # Just show first 3 batches
            break
    
    print("\n" + "=" * 80)
    print("Done! Check debug images at:", debug_dir)
    print("=" * 80)


def create_synthetic_manifest():
    """Create a synthetic manifest for testing."""
    import tempfile
    
    tmpdir = tempfile.mkdtemp()
    print(f"Creating synthetic data in: {tmpdir}")
    
    manifest = {}
    
    for i in range(5):
        subj_id = f"patient_{i:03d}"
        
        # Create synthetic image
        img_array = np.random.randn(32, 32, 32).astype(np.float32) * 100 + 500
        img_sitk = sitk.GetImageFromArray(img_array)
        img_sitk.SetSpacing([2.0, 2.0, 2.0])
        img_path = os.path.join(tmpdir, f"{subj_id}_image.nii.gz")
        sitk.WriteImage(img_sitk, img_path)
        
        # Create synthetic ROI
        roi_array = np.zeros((32, 32, 32), dtype=np.uint8)
        roi_array[10:22, 10:22, 10:22] = 1  # Simple box
        roi_sitk = sitk.GetImageFromArray(roi_array)
        roi_sitk.SetSpacing([2.0, 2.0, 2.0])
        roi_path = os.path.join(tmpdir, f"{subj_id}_roi.nii.gz")
        sitk.WriteImage(roi_sitk, roi_path)
        
        # Random angle
        angle = np.random.uniform(0, 180)
        
        manifest[subj_id] = {
            "original_image": img_path,
            "original_roi": roi_path,
            "angle": angle
        }
    
    manifest_path = os.path.join(tmpdir, "patient.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path


if __name__ == "__main__":
    main()
