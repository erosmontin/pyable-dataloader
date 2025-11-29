# PyAble DataLoader Architecture

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INPUT DATA                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Manifest (JSON/CSV)              Medical Images (NIfTI/DICOM)         │
│  ┌──────────────────┐             ┌──────────────┐                     │
│  │ subject_001:     │             │ T1.nii.gz    │                     │
│  │   images: [...]  │────────────>│ T2.nii.gz    │                     │
│  │   rois: [...]    │             │ FLAIR.nii.gz │                     │
│  │   label: 1.0     │             └──────────────┘                     │
│  └──────────────────┘             ┌──────────────┐                     │
│                                    │ roi.nii.gz   │                     │
│                                    └──────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PyableDataset.__getitem__(idx)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: Load with pyable                                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ img = SITKImaginable(filename)                                   │  │
│  │ roi = Roiable(filename)                                          │  │
│  │ img.resampleOnCanonicalSpace()  # Ensure LPS orientation        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 2: Select Reference Space                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ reference = select_reference(images)                             │  │
│  │   - First image                                                  │  │
│  │   - Largest image                                                │  │
│  │   - Custom function                                              │  │
│  │   - Global template                                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 3: Compute ROI Center (if needed)                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ roi_center = compute_roi_center(roi)                             │  │
│  │ # Physical coordinates (X, Y, Z) in mm                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 4: Create Centered Reference                                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ centered_ref = create_centered_reference(                        │  │
│  │     source_image,                                                │  │
│  │     roi_center,                                                  │  │
│  │     target_size,                                                 │  │
│  │     target_spacing                                               │  │
│  │ )                                                                │  │
│  │ # Centers volume around ROI to maximize information              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 5: Resample All Images                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ for img in images:                                               │  │
│  │     img.resampleOnTargetImage(centered_ref)                      │  │
│  │     # Linear interpolation for images                            │  │
│  │                                                                  │  │
│  │ for roi in rois:                                                 │  │
│  │     roi.resampleOnTargetImage(centered_ref)                      │  │
│  │     # Nearest-neighbor for labels (automatic!)                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 6: Convert to NumPy                                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ arrays = [img.getImageAsNumpy() for img in images]              │  │
│  │ # Returns (Z, Y, X) format - pyable v3 convention!              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 7: Apply ROI Masking (if enabled)                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ if mask_with_roi:                                                │  │
│  │     combined_mask = np.any([roi > 0 for roi in rois], axis=0)   │  │
│  │     images = [img * combined_mask for img in images]            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 8: Stack Channels (if enabled)                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ if stack_channels and len(images) > 1:                           │  │
│  │     images = np.stack(images, axis=0)  # C × Z × Y × X           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 9: Apply Transforms (if provided)                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ if transforms is not None:                                       │  │
│  │     images, rois, labelmaps = transforms(                        │  │
│  │         images, rois, labelmaps, meta                            │  │
│  │     )                                                            │  │
│  │     # IntensityNormalization, RandomFlip, etc.                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 10: Convert to PyTorch Tensors                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ images_tensor = torch.from_numpy(images).to(dtype)              │  │
│  │ roi_tensors = [torch.from_numpy(roi) for roi in rois]           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
└──────────────────────────────┼──────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Dictionary:                                                            │
│  {                                                                      │
│    'id': 'subject_001',                                                 │
│    'images': torch.Tensor (C × D × H × W),                              │
│    'rois': [torch.Tensor (D × H × W), ...],                             │
│    'labelmaps': [torch.Tensor (D × H × W), ...],                        │
│    'label': torch.Tensor (scalar),                                      │
│    'meta': {                                                            │
│      'spacing': [2.0, 2.0, 2.0],                                        │
│      'origin': [0.0, 0.0, 0.0],                                         │
│      'direction': (1,0,0, 0,1,0, 0,0,1),                                │
│      'size': [64, 64, 64],                                              │
│      'orientation': 'LPS',                                              │
│      'roi_center': [x, y, z],                                           │
│      'source_paths': {...}                                              │
│    }                                                                    │
│  }                                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PyTorch DataLoader                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Collates samples into batches:                                        │
│    - images: B × C × D × H × W                                          │
│    - labels: B                                                          │
│    - rois: List of B lists of tensors                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Your Training Loop                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  for batch in dataloader:                                              │
│      images = batch['images']                                           │
│      labels = batch['label']                                            │
│                                                                         │
│      outputs = model(images)                                            │
│      loss = criterion(outputs, labels)                                  │
│      loss.backward()                                                    │
│      optimizer.step()                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Transform Pipeline

```
Input: images (C×Z×Y×X), rois, labelmaps, meta
  │
  ▼
┌──────────────────────────────────┐
│  IntensityNormalization          │
│  - Z-score normalization         │
│  - Per-channel                   │
└──────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│  RandomFlip                      │
│  - Flip along specified axes     │
│  - Probability-based             │
└──────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│  RandomRotation90                │
│  - 90° rotations in axial plane  │
│  - Label-preserving              │
└──────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│  RandomAffine                    │
│  - Rotation, zoom, shift         │
│  - Nearest-neighbor for labels   │
└──────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│  RandomNoise                     │
│  - Gaussian noise                │
│  - Foreground only               │
└──────────────────────────────────┘
  │
  ▼
Output: transformed images, rois, labelmaps
```

## Caching System

```
First Access (no cache):
  Load image → Resample → Transform → Save to cache → Return
  Time: ~2.5s per sample

Subsequent Access (cached):
  Load from cache → Transform → Return
  Time: ~0.05s per sample

Cache Key Calculation:
  subject_id + file_mtimes + target_size + target_spacing + roi_settings
  → MD5 hash → cache/subject_001_abc123def456.npz

Cache Contents:
  - Resampled images (numpy array)
  - Resampled ROIs (numpy arrays)
  - Metadata (dict)
```

## Coordinate Systems

```
┌─────────────────────────────────────────────────────────────────┐
│                    Physical Space (mm)                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │  SimpleITK / ITK Coordinates                    │           │
│  │  (X, Y, Z) - Right, Anterior, Superior          │           │
│  │                                                 │           │
│  │  Spacing: (sx, sy, sz)                          │           │
│  │  Origin: (ox, oy, oz)                           │           │
│  │  Direction: 3×3 matrix                          │           │
│  └─────────────────────────────────────────────────┘           │
│                          │                                      │
│                          │ pyable conversion methods            │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────┐           │
│  │  NumPy Array Space (indices)                    │           │
│  │  (Z, Y, X) = (K, J, I) - Depth, Height, Width   │           │
│  │                                                 │           │
│  │  Shape: (D, H, W)                               │           │
│  │  pyable v3 convention!                          │           │
│  └─────────────────────────────────────────────────┘           │
│                          │                                      │
│                          │ torch.from_numpy()                   │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────┐           │
│  │  PyTorch Tensor                                 │           │
│  │  (C, D, H, W) - Channels, Depth, Height, Width  │           │
│  │                                                 │           │
│  │  Standard 3D medical image format               │           │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘

Conversion Functions (pyable v3):
  - getPhysicalPointFromArrayIndex(k, j, i) → (x, y, z)
  - getArrayIndexFromPhysicalPoint(x, y, z) → (k, j, i)
  - getImageAsNumpy() → returns (Z, Y, X) ✅
```

## Overlay Process

```
Model Output (C×D×H×W)
  │
  ▼ Remove batch/channel dims
Prediction Array (D×H×W) = (Z,Y,X)
  │
  ▼ Get overlayer function
overlayer = dataset.get_original_space_overlayer(subject_id)
  │
  ▼ Create SimpleITK image with resampled geometry
prediction_sitk = sitk.GetImageFromArray(prediction_array)
prediction_sitk.SetSpacing(resampled_spacing)
prediction_sitk.SetOrigin(resampled_origin)
prediction_sitk.SetDirection(resampled_direction)
  │
  ▼ Resample to original geometry
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(original_image)
resampler.SetInterpolator(sitk.sitkLinear)
  │
  ▼ Execute resampling
original_space_sitk = resampler.Execute(prediction_sitk)
  │
  ▼ Save or visualize
sitk.WriteImage(original_space_sitk, 'prediction.nii.gz')
```

## Class Hierarchy

```
torch.utils.data.Dataset
  │
  └── PyableDataset
        │
        ├── Uses: SITKImaginable (for images)
        │         Roiable (for ROIs)
        │         LabelMapable (for segmentations)
        │
        ├── Methods:
        │   ├── __init__()
        │   ├── __len__()
        │   ├── __getitem__()
        │   ├── get_original_space_overlayer()
        │   └── (internal helpers)
        │
        └── Features:
            ├── Manifest parsing (JSON/CSV)
            ├── Reference space selection
            ├── ROI-centered resampling
            ├── Label preservation
            ├── Caching
            └── Transform pipeline

MedicalImageTransform (base class)
  │
  ├── Compose
  ├── IntensityNormalization
  ├── RandomFlip
  ├── RandomRotation90
  ├── RandomAffine
  ├── RandomNoise
  └── CropOrPad
```

## File Organization

```
able-dataloader/
├── src/pyable_dataloader/
│   ├── __init__.py          # Exports
│   ├── dataset.py           # PyableDataset class
│   └── transforms.py        # Transform classes
│
├── tests/
│   └── test_basic.py        # Unit tests
│
├── examples/
│   └── example_*.py         # Usage examples
│
├── docs/                    # (future)
│
├── pyproject.toml           # Package config
├── README.md                # Main documentation
├── QUICKSTART.md            # Quick start guide
├── IMPLEMENTATION_SUMMARY.md # Implementation details
└── DATALOADER_BUILD_PROMPT.md # Build specification
```

---

This architecture ensures:
- ✅ Correct spatial transformations
- ✅ Label preservation
- ✅ Efficient caching
- ✅ Easy overlay of results
- ✅ Flexible and extensible design
