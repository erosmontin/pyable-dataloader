"""
PyTorch Dataset for Medical Images using pyable

This dataset provides:
1. Proper spatial transformations using pyable's Imaginable API
2. Support for multiple images per subject (stacked as channels)
3. ROI-based processing with label preservation
4. Registration transform integration
5. Caching for performance
6. Transform information for overlaying results back to original space
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Union, List, Dict, Callable, Optional, Tuple
import warnings

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    from pyable.imaginable import SITKImaginable, Roiable, LabelMapable
except ImportError:
    raise ImportError(
        "pyable not found. Please install it first:\n"
        "  cd pyable && pip install -e ."
    )


def orient_to_lps(sitk_image: sitk.Image, target_orientation: str = 'LPS') -> sitk.Image:
    """
    Reorient a SimpleITK image to the target orientation using DICOMOrientImageFilter.
    
    This is more reliable than pyable's resampleOnCanonicalSpace() which can 
    return zeros for some images.
    
    Args:
        sitk_image: Input SimpleITK image
        target_orientation: Target orientation string (default 'LPS')
    
    Returns:
        Reoriented SimpleITK image
    """
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(target_orientation)
    return orient_filter.Execute(sitk_image)


class PyableDataset(Dataset):
    """
    PyTorch Dataset for loading medical images via pyable.
    
    This dataset maintains proper spatial relationships between original images
    and model inputs, allowing easy overlay and reference back to original space.
    
    Key Features:
    - Uses pyable's resampling for proper spatial reference
    - Centers images in resampling volume to maximize information
    - Stores transformation matrices for overlaying results
    - Supports ROI positioning at consistent coordinates
    - Label-preserving resampling for ROIs/segmentations
    
    Args:
        manifest: Path to JSON/CSV file(s) or dict with structure:
            {
                "subject_id": {
                    "images": [path1, path2, ...],
                    "rois": [path1, ...],
                    "labelmaps": [path1, ...],
                    "reference": int or path (optional),
                    "transforms": [transform_paths] (optional)
                }
            }
        
        target_size: Target voxel dimensions [D, H, W] (e.g., [50, 50, 50])
        
        target_spacing: Target spacing in mm (float or [x,y,z])
        
        reference_selector: How to choose reference image per subject
            - 'first': Use first image
            - 'largest': Use image with largest volume
            - int: Use image at this index
            - callable: Function(images_list) -> int
        
        reference_space: Global reference space (optional)
            - str: Path to reference image file
            - dict: {'spacing': [x,y,z], 'size': [x,y,z], 'origin': [...], 'direction': [...]}
            If provided, ALL subjects resampled to this space
        
        roi_center_target: Target physical coordinates for ROI center (optional)
            If provided, images are shifted so ROI center is at these coordinates
        
        mask_with_roi: If True, multiply image by ROI mask
        
        roi_labels: List of label values to keep in ROI (None = keep all)
        
        transforms: Optional[Callable] = None,
            Applied after resampling to reference space on Imaginable objects
        
        stack_channels: If True, stack multiple images as channels (C × D × H × W)
        
        cache_dir: Optional directory to cache resampled results
        
        force_reload: Force reload from disk instead of using cache
        
        dtype: PyTorch dtype for output tensors
        
        return_meta: If True, include metadata dict in output
        
        orientation: Standard orientation code (default 'LPS')
    
    Returns:
        Dictionary containing:
            - 'id': Subject identifier
            - 'images': torch.Tensor (C × D × H × W or D × H × W)
            - 'rois': List[torch.Tensor] (each D × H × W)
            - 'labelmaps': List[torch.Tensor] (each D × H × W)
            - 'meta': dict with spacing, origin, direction, etc. (if return_meta=True)
    """
    
    def __init__(
        self,
        manifest: Union[str, dict, List[str]],
        target_size: List[int],
        target_spacing: Union[float, List[float]] = 1.0,
        reference_selector: Union[str, int, Callable] = 'first',
        reference_space: Optional[Union[str, dict]] = None,
        roi_center_target: Optional[List[float]] = None,
        mask_with_roi: bool = False,
        roi_labels: Optional[List[int]] = None,
        transforms: Optional[Callable] = None,
        stack_channels: bool = True,
        cache_dir: Optional[str] = None,
        force_reload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_meta: bool = True,
        orientation: str = 'LPS'
    ):
        self.target_size = target_size
        
        # Convert target_spacing to list if scalar
        if isinstance(target_spacing, (int, float)):
            self.target_spacing = [float(target_spacing)] * 3
        else:
            self.target_spacing = list(target_spacing)
        
        self.reference_selector = reference_selector
        self.reference_space = reference_space
        self.roi_center_target = roi_center_target
        self.mask_with_roi = mask_with_roi
        self.roi_labels = roi_labels
        self.transforms = transforms
        self.stack_channels = stack_channels
        self.cache_dir = cache_dir
        self.force_reload = force_reload
        self.dtype = dtype
        self.return_meta = return_meta
        self.orientation = orientation
        
        # Load manifest
        self.data = self._load_manifest(manifest)
        self.ids = list(self.data.keys())
        
        # Create global reference if specified
        self.global_reference = None
        if reference_space is not None:
            self.global_reference = self._create_global_reference(reference_space)
    
    def _load_manifest(self, manifest: Union[str, dict, List[str]]) -> dict:
        """Load manifest from JSON, CSV, or dict."""
        if isinstance(manifest, dict):
            return manifest
        
        if isinstance(manifest, str):
            if manifest.endswith('.json'):
                with open(manifest, 'r') as f:
                    return json.load(f)
            elif manifest.endswith('.csv'):
                return self._load_csv_manifest(manifest)
            else:
                raise ValueError(f"Unsupported manifest file type: {manifest}")
        
        if isinstance(manifest, list):
            # Multiple CSV files
            data = {}
            for csv_path in manifest:
                csv_data = self._load_csv_manifest(csv_path)
                # Merge with existing data
                for subj_id, subj_data in csv_data.items():
                    if subj_id not in data:
                        data[subj_id] = subj_data
                    else:
                        # Merge lists
                        for key in ['images', 'rois', 'labelmaps']:
                            if key in subj_data:
                                if key not in data[subj_id]:
                                    data[subj_id][key] = []
                                data[subj_id][key].extend(subj_data[key])
            return data
        
        raise ValueError(f"Unsupported manifest type: {type(manifest)}")
    
    def _load_csv_manifest(self, csv_path: str) -> dict:
        """Load manifest from CSV file."""
        df = pd.read_csv(csv_path)
        data = {}
        
        # Detect CSV format
        if 'id' in df.columns:
            # Format: id, image_paths, roi_paths, labelmap_paths, reference, ...
            for _, row in df.iterrows():
                subj_id = str(row['id'])
                
                if subj_id not in data:
                    data[subj_id] = {
                        'images': [],
                        'rois': [],
                        'labelmaps': [],
                        'reference': None
                    }
                
                # Parse image paths (could be JSON array string or single path)
                if 'image_paths' in row and pd.notna(row['image_paths']):
                    try:
                        paths = json.loads(row['image_paths'])
                        data[subj_id]['images'].extend(paths)
                    except:
                        data[subj_id]['images'].append(str(row['image_paths']))
                
                # Parse ROI paths
                if 'roi_paths' in row and pd.notna(row['roi_paths']):
                    try:
                        paths = json.loads(row['roi_paths'])
                        data[subj_id]['rois'].extend(paths)
                    except:
                        data[subj_id]['rois'].append(str(row['roi_paths']))
                
                # Parse labelmap paths
                if 'labelmap_paths' in row and pd.notna(row['labelmap_paths']):
                    try:
                        paths = json.loads(row['labelmap_paths'])
                        data[subj_id]['labelmaps'].extend(paths)
                    except:
                        data[subj_id]['labelmaps'].append(str(row['labelmap_paths']))
                
                # Reference
                if 'reference' in row and pd.notna(row['reference']):
                    ref_val = row['reference']
                    if isinstance(ref_val, bool):
                        if ref_val:
                            data[subj_id]['reference'] = len(data[subj_id]['images']) - 1
                    elif isinstance(ref_val, (int, float)):
                        data[subj_id]['reference'] = int(ref_val)
                    elif isinstance(ref_val, str) and ref_val != 'first':
                        data[subj_id]['reference'] = ref_val  # Path to reference
        else:
            # Simple format: first column is label, rest are image paths
            for idx, row in df.iterrows():
                subj_id = str(idx)
                data[subj_id] = {
                    'images': [],
                    'rois': [],
                    'labelmaps': [],
                    'label': float(row.iloc[0])  # First column is label
                }
                
                # Rest are image paths
                for val in row.iloc[1:]:
                    if pd.notna(val) and len(str(val)) > 3:
                        data[subj_id]['images'].append(str(val))
        
        return data
    
    def _create_global_reference(self, reference_space):
        """Create a global reference image."""
        if isinstance(reference_space, str):
            return SITKImaginable(filename=reference_space)
        
        if isinstance(reference_space, dict):
            # Create reference from parameters
            size = reference_space.get('size', self.target_size)
            spacing = reference_space.get('spacing', self.target_spacing)
            origin = reference_space.get('origin', [0.0, 0.0, 0.0])
            direction = reference_space.get('direction', (1,0,0, 0,1,0, 0,0,1))
            
            # Create SimpleITK image
            ref_sitk = sitk.Image(size, sitk.sitkFloat32)
            ref_sitk.SetSpacing(spacing)
            ref_sitk.SetOrigin(origin)
            ref_sitk.SetDirection(direction)
            
            # Wrap in SITKImaginable
            return SITKImaginable(image=ref_sitk)
        
        raise ValueError(f"Unsupported reference_space type: {type(reference_space)}")
    
    def _get_cache_key(self, subject_id: str, item: dict) -> str:
        """Generate deterministic cache key."""
        # Include subject_id, file paths, target params, ROI settings
        key_parts = [
            subject_id,
            str(self.target_size),
            str(self.target_spacing),
            str(self.mask_with_roi),
            str(self.roi_labels),
            str(self.roi_center_target),
            str(self.orientation)
        ]
        
        # Add file hashes
        for img_path in item.get('images', []):
            if os.path.exists(img_path):
                key_parts.append(str(os.path.getmtime(img_path)))
        
        for roi_path in item.get('rois', []):
            if os.path.exists(roi_path):
                key_parts.append(str(os.path.getmtime(roi_path)))
        
        # Hash the key
        key_str = '_'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, subject_id: str, item: dict) -> Path:
        """Get cache file path."""
        if self.cache_dir is None:
            return None
        
        cache_key = self._get_cache_key(subject_id, item)
        cache_path = Path(self.cache_dir) / f"{subject_id}_{cache_key}.npz"
        return cache_path
    
    def _load_from_cache(self, cache_path: Path) -> Optional[dict]:
        """Load preprocessed data from cache."""
        if cache_path is None or not cache_path.exists():
            return None
        
        try:
            cached = np.load(cache_path, allow_pickle=True)
            return {
                'images': cached['images'],
                'rois': [cached[f'roi_{i}'] for i in range(len(cached.files) - 2) 
                         if f'roi_{i}' in cached.files],
                'labelmaps': [cached[f'labelmap_{i}'] for i in range(len(cached.files) - 2)
                              if f'labelmap_{i}' in cached.files],
                'meta': cached['meta'].item()
            }
        except Exception as e:
            warnings.warn(f"Failed to load cache {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, cache_path: Path, images, rois, labelmaps, meta):
        """Save preprocessed data to cache."""
        if cache_path is None:
            return
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_dict = {
            'images': images,
            'meta': meta
        }
        
        for i, roi in enumerate(rois):
            save_dict[f'roi_{i}'] = roi
        
        for i, lm in enumerate(labelmaps):
            save_dict[f'labelmap_{i}'] = lm
        
        try:
            np.savez_compressed(cache_path, **save_dict)
        except Exception as e:
            warnings.warn(f"Failed to save cache {cache_path}: {e}")
    
    def _select_reference(self, images: List[SITKImaginable], item: dict) -> SITKImaginable:
        """Select reference image for subject."""
        # Check if reference specified in manifest
        if 'reference' in item and item['reference'] is not None:
            ref = item['reference']
            if isinstance(ref, int):
                return images[ref]
            elif isinstance(ref, str):
                # Path to reference image
                return SITKImaginable(filename=ref)
        
        # Use reference_selector
        if self.reference_selector == 'first':
            return images[0]
        elif self.reference_selector == 'largest':
            return max(images, key=lambda x: np.prod(x.getImageSize()))
        elif isinstance(self.reference_selector, int):
            return images[self.reference_selector]
        elif callable(self.reference_selector):
            idx = self.reference_selector(images)
            return images[idx]
        
        raise ValueError(f"Invalid reference_selector: {self.reference_selector}")
    
    def _compute_roi_center(self, roi: Union[Roiable, LabelMapable]) -> Optional[Tuple[float, float, float]]:
        """Compute physical center of ROI."""
        roi_array = roi.getImageAsNumpy()  # Returns (Z,Y,X) in v3
        
        # Filter to specific labels if requested
        if self.roi_labels is not None:
            mask = np.isin(roi_array, self.roi_labels)
        else:
            mask = roi_array > 0
        
        if not np.any(mask):
            return None
        
        # Compute center of mass in array indices (Z,Y,X)
        from scipy import ndimage
        com_zyx = ndimage.center_of_mass(mask)
        
        # Convert to physical coordinates (need to reverse to X,Y,Z for ITK)
        com_kji = (int(com_zyx[0]), int(com_zyx[1]), int(com_zyx[2]))
        physical_xyz = roi.getPhysicalPointFromArrayIndex(com_kji)
        
        return physical_xyz
    
    def _create_centered_reference(
        self,
        source_image: SITKImaginable,
        roi_center: Optional[Tuple[float, float, float]] = None
    ) -> SITKImaginable:
        """
        Create reference image centered around source or ROI.
        
        This ensures maximum information is preserved in the resampled volume.
        """
        # Calculate target physical size
        target_physical_size = np.array(self.target_size) * np.array(self.target_spacing)
        
        # Get source image center
        source_size = np.array(source_image.getImageSize())
        source_spacing = np.array(source_image.getImageSpacing())
        source_origin = np.array(source_image.getImageOrigin())
        
        # Calculate source physical center (in X,Y,Z)
        source_physical_center = source_origin + (source_size * source_spacing) / 2.0
        
        # Adjust center if ROI target is specified
        center_point = source_physical_center
        if self.roi_center_target is not None and roi_center is not None:
            # Shift so ROI center matches target
            roi_center = np.array(roi_center)
            target = np.array(self.roi_center_target)
            shift = target - roi_center
            center_point = source_physical_center + shift
        
        # Calculate new origin to center the volume
        new_origin = center_point - target_physical_size / 2.0
        
        # Create reference image
        ref_sitk = sitk.Image(self.target_size, sitk.sitkFloat32)
        ref_sitk.SetSpacing(self.target_spacing)
        ref_sitk.SetOrigin(new_origin.tolist())
        ref_sitk.SetDirection(source_image.getImageDirection())
        
        return SITKImaginable(image=ref_sitk)
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> dict:
        """Get preprocessed item."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        subject_id = self.ids[idx]
        item = self.data[subject_id]
        
        # Check cache
        cache_path = self._get_cache_path(subject_id, item)
        if not self.force_reload:
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                return self._format_output(subject_id, cached, item)
        
        # Load images
        images = [SITKImaginable(filename=p) for p in item.get('images', [])]
        if not images:
            raise ValueError(f"No images found for subject {subject_id}")
        
        # Standardize orientation to LPS using pyable's dicomOrient method
        for img in images:
            img.dicomOrient(self.orientation)
        
        # Load ROIs
        rois = []
        roi_center = None
        for roi_path in item.get('rois', []):
            roi = Roiable(filename=roi_path)
            roi.dicomOrient(self.orientation)
            
            # Filter labels if requested
            if self.roi_labels is not None:
                roi.filterValues(self.roi_labels)
            
            rois.append(roi)
            
            # Compute center if needed
            if roi_center is None and (self.roi_center_target is not None or self.mask_with_roi):
                roi_center = self._compute_roi_center(roi)
        
        # Load labelmaps
        labelmaps = []
        for lm_path in item.get('labelmaps', []):
            lm = LabelMapable(filename=lm_path)
            lm.dicomOrient(self.orientation)
            labelmaps.append(lm)
        
        # Determine reference space
        if self.global_reference is not None:
            reference = self.global_reference
        else:
            # Select per-subject reference
            ref_image = self._select_reference(images, item)
            # Create centered reference
            reference = self._create_centered_reference(ref_image, roi_center)
        
        # Resample all images to reference
        resampled_images = []
        for img in images:
            img_copy = SITKImaginable(image=sitk.Image(img.getImage()))
            img_copy.resampleOnTargetImage(reference)
            resampled_images.append(img_copy)
        
        # Resample ROIs
        resampled_rois = []
        for roi in rois:
            roi_copy = Roiable(image=sitk.Image(roi.getImage()))
            roi_copy.resampleOnTargetImage(reference)
            resampled_rois.append(roi_copy)
        
        # Resample labelmaps
        resampled_labelmaps = []
        for lm in labelmaps:
            lm_copy = LabelMapable(image=sitk.Image(lm.getImage()))
            lm_copy.resampleOnTargetImage(reference)
            resampled_labelmaps.append(lm_copy)
        
        # Collect metadata (needed by transforms)
        meta = {
            'subject_id': subject_id,
            'spacing': list(reference.getImageSpacing()),
            'origin': list(reference.getImageOrigin()),
            'direction': list(reference.getImageDirection()),
            'size': list(reference.getImageSize()),
        }
        
        # Add roi_center only if available (don't include None values which break collation)
        if roi_center is not None:
            meta['roi_center'] = roi_center
        
        # Add label if present in manifest
        if 'label' in item:
            meta['label'] = item['label']
        
        # Apply transforms if provided
        if self.transforms is not None:
            resampled_images, resampled_rois, resampled_labelmaps = self.transforms(resampled_images, resampled_rois, resampled_labelmaps, meta)
        
        # Convert to numpy arrays (ZYX format in v3) - handle both Imaginable and numpy array types
        def to_numpy(obj):
            """Convert object to numpy array, handling both Imaginable and numpy types."""
            if hasattr(obj, 'getImageAsNumpy'):
                return obj.getImageAsNumpy()
            elif isinstance(obj, np.ndarray):
                return obj
            else:
                raise TypeError(f"Cannot convert {type(obj)} to numpy array")
        
        image_arrays = [to_numpy(img) for img in resampled_images]
        roi_arrays = [to_numpy(roi) for roi in resampled_rois]
        labelmap_arrays = [to_numpy(lm) for lm in resampled_labelmaps]
        
        # Apply ROI masking if requested
        if self.mask_with_roi and roi_arrays:
            # Combine all ROI masks
            combined_mask = np.any([arr > 0 for arr in roi_arrays], axis=0)
            # Apply to each image
            image_arrays = [arr * combined_mask for arr in image_arrays]
        
        # Stack images if requested
        if self.stack_channels and len(image_arrays) > 1:
            images_array = np.stack(image_arrays, axis=0)  # C × Z × Y × X
        elif image_arrays:
            images_array = image_arrays[0]
        else:
            images_array = np.zeros(self.target_size)
        
        # Update metadata with additional fields
        meta.update({
            'orientation': reference.getOrientationCode(),
            'source_paths': {
                'images': item.get('images', []),
                'rois': item.get('rois', []),
                'labelmaps': item.get('labelmaps', [])
            }
        })
        
        # Save to cache
        self._save_to_cache(cache_path, images_array, roi_arrays, labelmap_arrays, meta)
        
        # Format output
        result = {
            'images': images_array,
            'rois': roi_arrays,
            'labelmaps': labelmap_arrays,
            'meta': meta
        }
        
        return self._format_output(subject_id, result, item)
    
    def _format_output(self, subject_id: str, data: dict, item: dict) -> dict:
        """Format output with proper tensor conversion."""
        images = data['images']
        rois = data['rois']
        labelmaps = data['labelmaps']
        meta = data['meta']
        
        # Convert to PyTorch tensors
        images_tensor = torch.from_numpy(images).to(self.dtype)
        
        # Ensure channel dimension exists
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)  # Add channel dim
        
        roi_tensors = [torch.from_numpy(roi).to(self.dtype) for roi in rois]
        labelmap_tensors = [torch.from_numpy(lm).to(self.dtype) for lm in labelmaps]
        
        result = {
            'id': subject_id,
            'images': images_tensor,
            'rois': roi_tensors,
            'labelmaps': labelmap_tensors
        }
        
        # Add label if present
        if 'label' in meta:
            result['label'] = torch.tensor(meta['label'], dtype=self.dtype)
        elif 'label' in item:
            result['label'] = torch.tensor(item['label'], dtype=self.dtype)
        
        if self.return_meta:
            result['meta'] = meta
        
        return result
    
    def get_original_space_overlayer(self, subject_id: str):
        """
        Get a function to overlay model outputs back to original image space.
        
        Args:
            subject_id: Subject identifier
        
        Returns:
            overlayer: Function(resampled_array) -> original_space_sitk_image
        """
        idx = self.ids.index(subject_id)
        item = self.data[subject_id]
        
        # Load original image
        original_path = item['images'][0]
        original = SITKImaginable(filename=original_path)
        
        # Get metadata for subject
        sample = self.__getitem__(idx)
        meta = sample['meta']
        
        def overlayer(resampled_array: np.ndarray, interpolator='linear') -> sitk.Image:
            """
            Overlay resampled array back to original space.
            
            Args:
                resampled_array: Array in resampled space (Z,Y,X)
                interpolator: 'linear', 'nearest', or 'bspline'
            
            Returns:
                SimpleITK image in original space
            """
            # Create SimpleITK image from array
            resampled_sitk = sitk.GetImageFromArray(resampled_array.astype(np.float32))
            resampled_sitk.SetSpacing(meta['spacing'])
            resampled_sitk.SetOrigin(meta['origin'])
            resampled_sitk.SetDirection(meta['direction'])
            
            # Resample to original space
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(original.getITKImage())
            
            if interpolator == 'nearest':
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            elif interpolator == 'bspline':
                resampler.SetInterpolator(sitk.sitkBSpline)
            else:
                resampler.SetInterpolator(sitk.sitkLinear)
            
            resampler.SetDefaultPixelValue(0)
            
            return resampler.Execute(resampled_sitk)
        
        return overlayer
