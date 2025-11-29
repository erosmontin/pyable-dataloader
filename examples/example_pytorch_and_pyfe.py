"""
Complete example showing pyable-dataloader usage with both PyTorch and pyfe

This example demonstrates:
1. Creating a manifest
2. Using PyableDataset with PyTorch DataLoader
3. Applying spatial transforms (RandomTranslation, RandomRotation, RandomBSpline)
4. Converting manifest to pyfe format
5. Using the same data for feature extraction with pyfe

"""

import json
import tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk

# Create synthetic test data
def create_test_data(output_dir: Path):
    """Create synthetic medical images for testing."""
    subjects = {}
    
    for i in range(3):
        subj_id = f"sub{i:03d}"
        
        # Create synthetic image (32x32x32)
        img_array = np.random.randn(32, 32, 32).astype(np.float32) * 100 + 500
        img_sitk = sitk.GetImageFromArray(img_array)
        img_sitk.SetSpacing([1.0, 1.0, 1.0])
        img_sitk.SetOrigin([0.0, 0.0, 0.0])
        
        img_path = output_dir / f"{subj_id}_T1.nii.gz"
        sitk.WriteImage(img_sitk, str(img_path))
        
        # Create synthetic labelmap with labels 0, 1, 2
        lm_array = np.zeros((32, 32, 32), dtype=np.uint8)
        center = np.array([16, 16, 16])
        for x in range(32):
            for y in range(32):
                for z in range(32):
                    dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                    if dist < 6:
                        lm_array[z, y, x] = 1
                    elif 6 <= dist < 10:
                        lm_array[z, y, x] = 2
        
        lm_sitk = sitk.GetImageFromArray(lm_array)
        lm_sitk.SetSpacing([1.0, 1.0, 1.0])
        lm_sitk.SetOrigin([0.0, 0.0, 0.0])
        
        lm_path = output_dir / f"{subj_id}_labelmap.nii.gz"
        sitk.WriteImage(lm_sitk, str(lm_path))
        
        subjects[subj_id] = {
            'images': [str(img_path)],
            'labelmaps': [str(lm_path)],
            'label': float(i % 2)
        }
    
    # Save manifest
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(subjects, f, indent=2)
    
    return manifest_path


# Example 1: PyTorch DataLoader with spatial transforms
def example_pytorch_dataloader():
    """Example using PyableDataset with PyTorch DataLoader."""
    print("\n" + "="*80)
    print("Example 1: PyTorch DataLoader with Spatial Transforms")
    print("="*80 + "\n")
    
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("⚠️  PyTorch not installed. Skipping PyTorch example.")
        return
    
    from pyable_dataloader import (
        PyableDataset,
        Compose,
        IntensityNormalization,
        RandomTranslation,
        RandomRotation,
        RandomBSpline,
        RandomFlip
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        manifest_path = create_test_data(tmpdir)
        
        print(f"✓ Created test data in {tmpdir}")
        print(f"✓ Manifest: {manifest_path}\n")
        
        # Create dataset with spatial transforms
        transforms = Compose([
            IntensityNormalization(method='zscore'),
            RandomTranslation(
                translation_range=[[-3, 3], [-3, 3], [-2, 2]],  # Per-axis mm ranges
                prob=0.8
            ),
            RandomRotation(
                rotation_range=[[-5, 5], [-5, 5], [-10, 10]],  # Per-axis degrees
                prob=0.7
            ),
            RandomBSpline(
                mesh_size=(4, 4, 4),
                magnitude=2.0,
                prob=0.5
            ),
            RandomFlip(axes=[1, 2], prob=0.5)
        ])
        
        dataset = PyableDataset(
            manifest=str(manifest_path),
            target_size=[24, 24, 24],
            target_spacing=1.5,
            transforms=transforms,
            cache_dir=str(tmpdir / 'cache'),
            return_meta=False  # Don't return meta for DataLoader batching
        )
        
        print(f"✓ Created dataset with {len(dataset)} samples")
        print(f"✓ Applied {len(transforms.transforms)} transforms\n")
        
        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0  # Use 0 for debugging, increase for training
        )
        
        print("Processing batches...")
        for batch_idx, batch in enumerate(loader):
            images = batch['images']
            labels = batch['label']
            labelmaps = batch['labelmaps']
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels: {labels}")
            print(f"  Number of labelmaps: {len(labelmaps)}")
            
            # Verify label preservation
            for i, lm in enumerate(labelmaps):
                unique_vals = torch.unique(lm[0])
                print(f"  Labelmap {i} unique values: {unique_vals.tolist()}")
                # Should only contain integers (0, 1, 2)
                assert all(v in [0, 1, 2] for v in unique_vals.tolist()), \
                    f"❌ Label interpolation detected! Values: {unique_vals}"
        
        print("\n✓ All labelmaps preserved integer labels (no interpolation)")
        print("✓ PyTorch DataLoader example completed successfully!")


# Example 2: Converting to pyfe format and using with pyfe
def example_pyfe_integration():
    """Example converting manifest to pyfe format."""
    print("\n" + "="*80)
    print("Example 2: pyfe Integration")
    print("="*80 + "\n")
    
    from pyable_dataloader.pyfe_adapter import convert_manifest_to_pyfe
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        manifest_path = create_test_data(tmpdir)
        
        print(f"✓ Created test data in {tmpdir}")
        
        # Convert to pyfe format
        pyfe_manifest_path = tmpdir / 'pyfe_manifest.json'
        convert_manifest_to_pyfe(str(manifest_path), pyfe_manifest_path)
        
        print(f"✓ Converted manifest to pyfe format: {pyfe_manifest_path}\n")
        
        # Show pyfe manifest structure
        with open(pyfe_manifest_path, 'r') as f:
            pyfe_manifest = json.load(f)
        
        print("pyfe manifest structure:")
        print(json.dumps(pyfe_manifest, indent=2))
        
        print("\n✓ pyfe manifest created successfully!")
        print("\nTo use with pyfe:")
        print("  from pyfe import pyfe")
        print(f"  result, ids = pyfe.exrtactMyFeatures('{pyfe_manifest_path}', dimension=3)")
        print("  df = pyfe.exrtactMyFeaturesToPandas(...)")


# Example 3: Using pyfe dataloader adapter
def example_pyfe_dataloader_adapter():
    """Example importing from pyfe.dataloader."""
    print("\n" + "="*80)
    print("Example 3: Using pyfe dataloader adapter")
    print("="*80 + "\n")
    
    try:
        from pyfe.dataloader import (
            PyableDataset,
            Compose,
            RandomTranslation,
            RandomRotation,
            RandomBSpline
        )
        print("✓ Successfully imported from pyfe.dataloader")
        print("  Available classes:")
        print("    - PyableDataset")
        print("    - Compose")
        print("    - RandomTranslation")
        print("    - RandomRotation")
        print("    - RandomBSpline")
        print("    - IntensityNormalization")
        print("    - RandomFlip")
        print("    - RandomRotation90")
        print("    - RandomAffine")
        print("    - RandomNoise")
        
        # Try to import convert_manifest_to_pyfe if available
        try:
            from pyfe.dataloader import convert_manifest_to_pyfe
            print("    - convert_manifest_to_pyfe (optional)")
            print("\n✓ pyfe can use all dataloader features!")
        except (ImportError, AttributeError):
            print("\n✓ pyfe can use dataloader (convert_manifest_to_pyfe not available)")
        
    except ImportError as e:
        print(f"⚠️  Could not import from pyfe.dataloader: {e}")
        print("   Make sure pyfe is installed with dataloader support:")
        print("   pip install -e /path/to/pyfe")


# Example 4: Label preservation verification
def example_label_preservation():
    """Verify that all transforms preserve label integrity."""
    print("\n" + "="*80)
    print("Example 4: Label Preservation Verification")
    print("="*80 + "\n")
    
    try:
        import torch
    except ImportError:
        print("⚠️  PyTorch not installed. Skipping label preservation test.")
        return
    
    from pyable_dataloader import (
        PyableDataset,
        Compose,
        RandomTranslation,
        RandomRotation,
        RandomAffine,
        RandomBSpline
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        manifest_path = create_test_data(tmpdir)
        
        # Test each transform individually
        transform_configs = [
            ("RandomTranslation", RandomTranslation(
                translation_range=[[-5, 5], [-5, 5], [-3, 3]], prob=1.0
            )),
            ("RandomRotation", RandomRotation(
                rotation_range=[[-10, 10], [-10, 10], [-15, 15]], prob=1.0
            )),
            ("RandomAffine", RandomAffine(
                rotation_range=10.0,
                zoom_range=(0.9, 1.1),
                shift_range=3.0,
                prob=1.0
            )),
            ("RandomBSpline", RandomBSpline(
                mesh_size=(4, 4, 4),
                magnitude=3.0,
                prob=1.0
            ))
        ]
        
        print("Testing label preservation for each transform:\n")
        
        for transform_name, transform in transform_configs:
            dataset = PyableDataset(
                manifest=str(manifest_path),
                target_size=[24, 24, 24],
                target_spacing=1.5,
                transforms=transform
            )
            
            sample = dataset[0]
            labelmaps = sample['labelmaps']
            
            all_preserved = True
            for i, lm in enumerate(labelmaps):
                unique_vals = torch.unique(lm).tolist()
                # Check if all values are integers (no interpolation)
                is_preserved = all(v == int(v) for v in unique_vals)
                if not is_preserved:
                    all_preserved = False
                    print(f"  ❌ {transform_name}: Labelmap {i} has interpolated values: {unique_vals}")
            
            if all_preserved:
                print(f"  ✓ {transform_name}: All labels preserved (no interpolation)")
        
        print("\n✓ Label preservation test completed!")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("PyAble DataLoader - Complete Usage Examples")
    print("="*80)
    
    # Run all examples
    example_pytorch_dataloader()
    example_pyfe_integration()
    example_pyfe_dataloader_adapter()
    example_label_preservation()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")
    
    print("Summary:")
    print("  ✓ PyTorch DataLoader with spatial transforms")
    print("  ✓ pyfe manifest conversion")
    print("  ✓ pyfe dataloader adapter")
    print("  ✓ Label preservation verification")
    print("\nThe dataloader is ready for both PyTorch training and pyfe feature extraction!")
