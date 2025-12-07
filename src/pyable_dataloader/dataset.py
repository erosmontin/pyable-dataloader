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
from typing import Union, List, Dict, Callable, Optional, Tuple, Any
import random
import warnings
import tempfile

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
        
        debug_save_dir: Optional directory to save processed images for debugging
        
        debug_save_format: Format for debug saves ('nifti' or 'numpy', default 'nifti')
    
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
        label_dtype: Optional[torch.dtype] = None,
        return_meta: bool = True,
        orientation: str = 'LPS',
        debug_save_dir: Optional[str] = None,
        debug_save_format: str = 'nifti',
        return_numpy: bool = False,
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
        self.debug_save_dir = debug_save_dir
        self.debug_save_format = debug_save_format
        # If True, convenience getters will return numpy arrays instead of torch tensors
        self.return_numpy = return_numpy
        # Label dtype defaults to long (for classification targets). If None, infer later.
        self.label_dtype = label_dtype if label_dtype is not None else torch.int64
        
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
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        warnings.warn(
                            f"CSV manifest: subject {subj_id}: failed to decode 'image_paths' as JSON; "
                            f"treating value as single path: {e}"
                        )
                        data[subj_id]['images'].append(str(row['image_paths']))
                
                # Parse ROI paths
                if 'roi_paths' in row and pd.notna(row['roi_paths']):
                    try:
                        paths = json.loads(row['roi_paths'])
                        data[subj_id]['rois'].extend(paths)
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        warnings.warn(
                            f"CSV manifest: subject {subj_id}: failed to decode 'roi_paths' as JSON; "
                            f"treating value as single path: {e}"
                        )
                        data[subj_id]['rois'].append(str(row['roi_paths']))
                
                # Parse labelmap paths
                if 'labelmap_paths' in row and pd.notna(row['labelmap_paths']):
                    try:
                        paths = json.loads(row['labelmap_paths'])
                        data[subj_id]['labelmaps'].extend(paths)
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        warnings.warn(
                            f"CSV manifest: subject {subj_id}: failed to decode 'labelmap_paths' as JSON; "
                            f"treating value as single path: {e}"
                        )
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
            # Explicitly collect ROI and labelmap keys to avoid fragile index logic
            roi_keys = sorted([k for k in cached.files if k.startswith('roi_')])
            labelmap_keys = sorted([k for k in cached.files if k.startswith('labelmap_')])
            rois = [cached[k] for k in roi_keys]
            labelmaps = [cached[k] for k in labelmap_keys]
            return {
                'images': cached['images'],
                'rois': rois,
                'labelmaps': labelmaps,
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
            # Write to temp file and atomically replace to avoid partial writes
            tmp_path = cache_path.with_suffix('.tmp')
            np.savez_compressed(tmp_path, **save_dict)
            if tmp_path.exists():
                os.replace(tmp_path, cache_path)
            else:
                raise IOError(f"Temp cache file not created: {tmp_path}")
        except Exception as e:
            # Try to remove the partially written tmp file if present
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            warnings.warn(f"Failed to save cache {cache_path}: {e}")
    
    def _save_debug_images(self, subject_id: str, image_arrays: List[np.ndarray], 
                          roi_arrays: List[np.ndarray], labelmap_arrays: List[np.ndarray], 
                          meta: dict):
        """Save processed images for debugging."""
        if self.debug_save_dir is None:
            return
        
        debug_dir = Path(self.debug_save_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images
        for i, img_array in enumerate(image_arrays):
            if self.debug_save_format == 'nifti':
                # Convert numpy array back to SimpleITK image
                sitk_img = sitk.GetImageFromArray(img_array.astype(np.float32))
                sitk_img.SetSpacing(meta['spacing'])
                sitk_img.SetOrigin(meta['origin'])
                sitk_img.SetDirection(meta['direction'])
                
                filename = debug_dir / f"{subject_id}_image_{i}.nii.gz"
                sitk.WriteImage(sitk_img, str(filename))
            elif self.debug_save_format == 'numpy':
                filename = debug_dir / f"{subject_id}_image_{i}.npy"
                np.save(filename, img_array)
        
        # Save ROIs
        for i, roi_array in enumerate(roi_arrays):
            if self.debug_save_format == 'nifti':
                sitk_roi = sitk.GetImageFromArray(roi_array.astype(np.uint8))
                sitk_roi.SetSpacing(meta['spacing'])
                sitk_roi.SetOrigin(meta['origin'])
                sitk_roi.SetDirection(meta['direction'])
                
                filename = debug_dir / f"{subject_id}_roi_{i}.nii.gz"
                sitk.WriteImage(sitk_roi, str(filename))
            elif self.debug_save_format == 'numpy':
                filename = debug_dir / f"{subject_id}_roi_{i}.npy"
                np.save(filename, roi_array)
        
        # Save labelmaps
        for i, lm_array in enumerate(labelmap_arrays):
            if self.debug_save_format == 'nifti':
                sitk_lm = sitk.GetImageFromArray(lm_array.astype(np.uint8))
                sitk_lm.SetSpacing(meta['spacing'])
                sitk_lm.SetOrigin(meta['origin'])
                sitk_lm.SetDirection(meta['direction'])
                
                filename = debug_dir / f"{subject_id}_labelmap_{i}.nii.gz"
                sitk.WriteImage(sitk_lm, str(filename))
            elif self.debug_save_format == 'numpy':
                filename = debug_dir / f"{subject_id}_labelmap_{i}.npy"
                np.save(filename, lm_array)
        
    def get_numpy_item(self, idx: int, as_nifti: bool = False, as_pyable: bool = False, transforms: Optional[Callable] = None, save_to_files: bool = False) -> dict:
        """Return the pre-tensor data for a sample.

        Args:
            idx: Dataset index
            as_nifti: If True, return SimpleITK images for images/rois/labelmaps
            as_pyable: If True, return pyable `SITKImaginable`, `Roiable`, and `LabelMapable` objects
            transforms: Optional additional transforms to apply (e.g., for on-demand augmentation)
            save_to_files: If True, save processed arrays to temporary NIfTI files and return file paths

        Returns:
            dict with keys: 'images' (list of arrays, images, or paths), 'rois', 'labelmaps', 'meta'
        """
        # Use the normal __getitem__ pipeline and convert back to numpy if needed
        # Insert the src path to ensure local code is used when testing via src/
        sample = None
        # Call internal pipeline by getting the processed result (before tensor conversion)
        # The existing implementation builds a `result` dict before calling _format_output.
        # To avoid duplicating logic, we'll call __getitem__ and convert tensors back to numpy.
        item = self.__getitem__(idx)

        # Extract images
        images = item.get('images')
        rois = item.get('rois', [])
        labelmaps = item.get('labelmaps', [])
        meta = item.get('meta', {})

        # Convert torch tensors to numpy
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return x

        images_np = to_numpy(images)
        # images_np may be CxZxYxX or ZxYxX
        if isinstance(images_np, np.ndarray) and images_np.ndim == 4:
            image_arrays = [images_np[i] for i in range(images_np.shape[0])]
        elif isinstance(images_np, np.ndarray) and images_np.ndim == 3:
            image_arrays = [images_np]
        else:
            image_arrays = images_np if isinstance(images_np, list) else [images_np]

        roi_arrays = [to_numpy(r) for r in rois]
        labelmap_arrays = [to_numpy(lm) for lm in labelmaps]

        # Apply additional transforms if provided (e.g., on-demand augmentation)
        if transforms is not None:
            image_arrays, roi_arrays, labelmap_arrays = transforms(image_arrays, roi_arrays, labelmap_arrays, meta)

        # Optionally save to temporary files and return paths
        if save_to_files:
            temp_dir = Path(tempfile.mkdtemp())
            
            image_paths = []
            for i, arr in enumerate(image_arrays):
                sitk_img = sitk.GetImageFromArray(arr.astype(np.float32))
                if 'spacing' in meta:
                    sitk_img.SetSpacing(meta['spacing'])
                if 'origin' in meta:
                    sitk_img.SetOrigin(meta['origin'])
                if 'direction' in meta:
                    sitk_img.SetDirection(meta['direction'])
                path = temp_dir / f"image_{i}.nii.gz"
                sitk.WriteImage(sitk_img, str(path))
                image_paths.append(str(path))
            
            roi_paths = []
            for i, arr in enumerate(roi_arrays):
                sitk_roi = sitk.GetImageFromArray(arr.astype(np.uint8))
                if 'spacing' in meta:
                    sitk_roi.SetSpacing(meta['spacing'])
                if 'origin' in meta:
                    sitk_roi.SetOrigin(meta['origin'])
                if 'direction' in meta:
                    sitk_roi.SetDirection(meta['direction'])
                path = temp_dir / f"roi_{i}.nii.gz"
                sitk.WriteImage(sitk_roi, str(path))
                roi_paths.append(str(path))
            
            labelmap_paths = []
            for i, arr in enumerate(labelmap_arrays):
                sitk_lm = sitk.GetImageFromArray(arr.astype(np.uint8))
                if 'spacing' in meta:
                    sitk_lm.SetSpacing(meta['spacing'])
                if 'origin' in meta:
                    sitk_lm.SetOrigin(meta['origin'])
                if 'direction' in meta:
                    sitk_lm.SetDirection(meta['direction'])
                path = temp_dir / f"labelmap_{i}.nii.gz"
                sitk.WriteImage(sitk_lm, str(path))
                labelmap_paths.append(str(path))
            
            # Store temp_dir in meta for cleanup reference
            meta['temp_dir'] = str(temp_dir)
            
            return {
                'images': image_paths,
                'rois': roi_paths,
                'labelmaps': labelmap_paths,
                'meta': meta
            }

        # Optionally convert to SimpleITK images
        if as_nifti:
            sitk_images = []
            for arr in image_arrays:
                sitk_img = sitk.GetImageFromArray(arr.astype(np.float32))
                if 'spacing' in meta:
                    sitk_img.SetSpacing(meta['spacing'])
                if 'origin' in meta:
                    sitk_img.SetOrigin(meta['origin'])
                if 'direction' in meta:
                    sitk_img.SetDirection(meta['direction'])
                sitk_images.append(sitk_img)

            sitk_rois = []
            for arr in roi_arrays:
                sitk_roi = sitk.GetImageFromArray(arr.astype(np.uint8))
                if 'spacing' in meta:
                    sitk_roi.SetSpacing(meta['spacing'])
                if 'origin' in meta:
                    sitk_roi.SetOrigin(meta['origin'])
                if 'direction' in meta:
                    sitk_roi.SetDirection(meta['direction'])
                sitk_rois.append(sitk_roi)

            sitk_labelmaps = []
            for arr in labelmap_arrays:
                sitk_lm = sitk.GetImageFromArray(arr.astype(np.uint8))
                if 'spacing' in meta:
                    sitk_lm.SetSpacing(meta['spacing'])
                if 'origin' in meta:
                    sitk_lm.SetOrigin(meta['origin'])
                if 'direction' in meta:
                    sitk_lm.SetDirection(meta['direction'])
                sitk_labelmaps.append(sitk_lm)

            return {
                'images': sitk_images,
                'rois': sitk_rois,
                'labelmaps': sitk_labelmaps,
                'meta': meta
            }

        # Optionally convert to pyable objects
        if as_pyable:
            py_images = []
            for arr in image_arrays:
                sitk_img = sitk.GetImageFromArray(arr.astype(np.float32))
                if 'spacing' in meta:
                    sitk_img.SetSpacing(meta['spacing'])
                if 'origin' in meta:
                    sitk_img.SetOrigin(meta['origin'])
                if 'direction' in meta:
                    sitk_img.SetDirection(meta['direction'])
                py_images.append(SITKImaginable(image=sitk_img))

            py_rois = []
            for arr in roi_arrays:
                sitk_roi = sitk.GetImageFromArray(arr.astype(np.uint8))
                if 'spacing' in meta:
                    sitk_roi.SetSpacing(meta['spacing'])
                if 'origin' in meta:
                    sitk_roi.SetOrigin(meta['origin'])
                if 'direction' in meta:
                    sitk_roi.SetDirection(meta['direction'])
                py_rois.append(Roiable(image=sitk_roi))

            py_labelmaps = []
            for arr in labelmap_arrays:
                sitk_lm = sitk.GetImageFromArray(arr.astype(np.uint8))
                if 'spacing' in meta:
                    sitk_lm.SetSpacing(meta['spacing'])
                if 'origin' in meta:
                    sitk_lm.SetOrigin(meta['origin'])
                if 'direction' in meta:
                    sitk_lm.SetDirection(meta['direction'])
                py_labelmaps.append(LabelMapable(image=sitk_lm))

            return {
                'images': py_images,
                'rois': py_rois,
                'labelmaps': py_labelmaps,
                'meta': meta
            }

        # Default: return numpy arrays
        return {
            'images': image_arrays,
            'rois': roi_arrays,
            'labelmaps': labelmap_arrays,
            'meta': meta
        }
    
    def get_multiple_augmentations(
        self,
        subject_idx: int,
        augmentation_configs: List[Dict[str, Any]],
        as_nifti: bool = False,
        save_to_files: bool = False,
        base_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple augmented versions of a sample using different transformation configs.

        Args:
            subject_idx: Index of subject in dataset
            augmentation_configs: List of dicts, each containing:
                - 'transforms': Compose object with augmentation pipeline
                - 'name': str identifier for this augmentation type
                - 'params': dict of parameters (optional, for logging)
            as_nifti: Return SimpleITK images instead of numpy arrays
            save_to_files: Save to temporary NIfTI files and return paths
            base_seed: Base random seed (will be modified per augmentation)

        Returns:
            List of dicts, one per augmentation config, each containing:
            - 'name': augmentation identifier
            - 'images': list of arrays/images/paths
            - 'rois': list of arrays/images/paths
            - 'labelmaps': list of arrays/images/paths
            - 'meta': metadata dict
            - 'config': original config dict
        """
        results = []
        for idx, config in enumerate(augmentation_configs):
            # Save RNG state and set reproducible seeds for numpy, torch and random
            old_np_state = np.random.get_state()
            old_random_state = random.getstate()
            old_torch_state = torch.get_rng_state()
            try:
                seed = int(base_seed) + int(idx)
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)

                # Get augmented sample
                sample = self.get_numpy_item(
                    subject_idx,
                    transforms=config.get('transforms'),
                    as_nifti=as_nifti,
                    save_to_files=save_to_files
                )
            finally:
                # Restore RNG state to avoid side effects
                np.random.set_state(old_np_state)
                random.setstate(old_random_state)
                torch.set_rng_state(old_torch_state)
            
            # Format result
            result = {
                'name': config['name'],
                'images': sample['images'],
                'rois': sample['rois'],
                'labelmaps': sample['labelmaps'],
                'meta': sample['meta'],
                'config': config
            }
            results.append(result)
        
        return results
    
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
        # 1) Try to use pyable-provided centroid helpers if present
        try:
            if hasattr(roi, "getCenterOfGravityCoordinates"):
                coords = roi.getCenterOfGravityCoordinates()
                if coords is not None:
                    return coords
            if hasattr(roi, "getCentroidCoordinates"):
                coords = roi.getCentroidCoordinates()
                if coords is not None:
                    return coords
        except Exception:
            # Silently fall back to the classic approach below if pyable helper fails
            pass

        # 2) Fallback to numpy/scipy-based centroid computation of the label mask
        roi_array = roi.getImageAsNumpy()  # Returns (Z,Y,X) in v3

        # Filter to specific labels if requested
        if self.roi_labels is not None:
            mask = np.isin(roi_array, self.roi_labels)
        else:
            mask = roi_array > 0

        if not np.any(mask):
            return None

        # Prefer SciPy center_of_mass if available (more robust), else fallback to a numpy implementation
        try:
            from scipy import ndimage
            com_zyx = ndimage.center_of_mass(mask)
        except Exception:
            coords = np.argwhere(mask)
            com_zyx = tuple(coords.mean(axis=0).tolist())

        # Convert to nearest integer array index and convert to physical coordinate
        com_kji = (int(np.round(com_zyx[0])), int(np.round(com_zyx[1])), int(np.round(com_zyx[2])))
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
        
        # Standardize orientation
        for img in images:
            img.resampleOnCanonicalSpace()  # Ensures LPS axis-aligned
        
        # Load ROIs
        rois = []
        roi_center = None
        for roi_path in item.get('rois', []):
            roi = Roiable(filename=roi_path)
            roi.resampleOnCanonicalSpace()
            
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
            lm.resampleOnCanonicalSpace()
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
        
        # Collect metadata
        meta = {
            'subject_id': subject_id,
            'spacing': reference.getImageSpacing(),
            'origin': reference.getImageOrigin(),
            'direction': reference.getImageDirection(),
            'size': reference.getImageSize(),
            'orientation': reference.getOrientationCode(),
            'source_paths': {
                'images': item.get('images', []),
                'rois': item.get('rois', []),
                'labelmaps': item.get('labelmaps', [])
            }
        }
        if roi_center is not None:
            meta['roi_center'] = roi_center
        
        # Add label if present
        if 'label' in item:
            meta['label'] = item['label']
        
        # Convert to numpy arrays (ZYX format in v3)
        image_arrays = [img.getImageAsNumpy() for img in resampled_images]
        roi_arrays = [roi.getImageAsNumpy() for roi in resampled_rois]
        labelmap_arrays = [lm.getImageAsNumpy() for lm in resampled_labelmaps]
        
        # Apply transforms if provided (before stacking)
        if self.transforms is not None:
            # For transforms, we need to handle the case where images might be stacked or not
            # Pass the list of arrays and let transforms handle stacking/unstacking
            image_arrays, roi_arrays, labelmap_arrays = self.transforms(image_arrays, roi_arrays, labelmap_arrays, meta)
        
        # Apply ROI masking if requested
        if self.mask_with_roi and roi_arrays:
            # Combine all ROI masks
            combined_mask = np.any([arr > 0 for arr in roi_arrays], axis=0)
            # Validate shapes
            for i, arr in enumerate(image_arrays):
                if combined_mask.shape != arr.shape:
                    raise ValueError(
                        f"ROI mask shape {combined_mask.shape} does not match image shape {arr.shape}"
                    )
            # Apply to each image
            image_arrays = [arr * combined_mask for arr in image_arrays]
        
        # Stack images if requested
        if self.stack_channels and len(image_arrays) > 1:
            images_array = np.stack(image_arrays, axis=0)  # C × Z × Y × X
        elif image_arrays:
            images_array = image_arrays[0]
        else:
            images_array = np.zeros(self.target_size)
        
        # Save debug images if requested
        self._save_debug_images(subject_id, image_arrays, roi_arrays, labelmap_arrays, meta)
        
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
        
        # If requested, return numpy arrays directly and skip torch conversion
        if self.return_numpy:
            result = {
                'id': subject_id,
                'images': images,
                'rois': rois,
                'labelmaps': labelmaps
            }
            if 'label' in meta:
                if self.label_dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                    # Convert to a plain int for numpy output
                    try:
                        result['label'] = int(meta['label'])
                    except Exception:
                        result['label'] = meta['label']
                else:
                    result['label'] = meta['label']
            elif 'label' in item:
                if self.label_dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                    try:
                        result['label'] = int(item['label'])
                    except Exception:
                        result['label'] = item['label']
                else:
                    result['label'] = item['label']
            if self.return_meta:
                result['meta'] = meta
            return result

        # Convert to PyTorch tensors
        images_tensor = torch.from_numpy(images).to(self.dtype)
        
        # Ensure channel dimension exists
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)  # Add channel dim
        
        # Convert ROIs/labelmaps to integer dtype by default
        roi_tensors = [torch.from_numpy(roi).to(torch.int64) for roi in rois]
        labelmap_tensors = [torch.from_numpy(lm).to(torch.int64) for lm in labelmaps]
        
        result = {
            'id': subject_id,
            'images': images_tensor,
            'rois': roi_tensors,
            'labelmaps': labelmap_tensors
        }
        
        # Add label if present
        if 'label' in meta:
            # If label_dtype is an integer dtype, coerce to int
            # Always cast to label_dtype when label_dtype is provided
            if self.label_dtype is not None:
                try:
                    # Convert floats to integers if label_dtype is integer type
                    val = meta['label']
                    if isinstance(val, (float, np.floating)) and self.label_dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                        val = int(val)
                    result['label'] = torch.tensor(val, dtype=self.label_dtype)
                except Exception:
                    # fallback
                    result['label'] = torch.tensor(meta['label'], dtype=self.dtype)
        elif 'label' in item:
            if self.label_dtype is not None:
                try:
                    val = item['label']
                    if isinstance(val, (float, np.floating)) and self.label_dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                        val = int(val)
                    result['label'] = torch.tensor(val, dtype=self.label_dtype)
                except Exception:
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


class MultipleAugmentationDataset(Dataset):
    """Dataset that generates multiple augmentations per sample using a base PyableDataset.

    It uses PyableDataset.get_multiple_augmentations to precompute augmented samples and
    supplies them as Dataset elements that are compatible with PyTorch DataLoader.
    """

    def __init__(self, base_dataset: PyableDataset, augmentation_configs: List[Dict[str, Any]], base_seed: int = 42):
        self.base_dataset = base_dataset
        self.augmentation_configs = augmentation_configs
        self.base_seed = base_seed
        self.augmented_data = []

        # Pre-generate all augmentations
        for subj_idx in range(len(base_dataset)):
            augmented_samples = base_dataset.get_multiple_augmentations(
                subject_idx=subj_idx,
                augmentation_configs=augmentation_configs,
                base_seed=base_seed
            )
            self.augmented_data.extend(augmented_samples)

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx: int):
        item = self.augmented_data[idx]

        # Convert to tensors by reusing PyableDataset._format_output behavior
        # We'll construct a lightweight dict and pass through _format_output to keep semantics.
        # Build pseudo item and sample dict
        images = item['images']
        rois = item['rois']
        labelmaps = item['labelmaps']
        meta = item['meta']
        # If images are numpy arrays, convert to PyTorch tensors
        if isinstance(images, np.ndarray):
            images_tensor = torch.from_numpy(images).to(self.base_dataset.dtype)
            if images_tensor.ndim == 3:
                images_tensor = images_tensor.unsqueeze(0)
        else:
            images_tensor = images

        roi_tensors = [torch.from_numpy(r).to(torch.int64) for r in rois]
        labelmap_tensors = [torch.from_numpy(lm).to(torch.int64) for lm in labelmaps]

        result = {
            'id': meta.get('subject_id', f'aug_{idx}'),
            'images': images_tensor,
            'rois': roi_tensors,
            'labelmaps': labelmap_tensors,
            'meta': meta,
            'augmentation_name': item.get('name'),
            'config': item.get('config')
        }

        if 'label' in meta:
            if isinstance(meta['label'], (int, np.integer)):
                result['label'] = torch.tensor(meta['label'], dtype=self.base_dataset.label_dtype)
            else:
                result['label'] = torch.tensor(meta['label'], dtype=self.base_dataset.dtype)

        return result


class MultipleAugmentationDataset(Dataset):
    """
    Dataset wrapper that generates multiple augmented versions of each sample.
    
    This allows using PyTorch DataLoader to batch multiple augmentations of the same
    subject, enabling robust training and feature stability analysis.
    
    Args:
        base_dataset: PyableDataset instance to wrap
        augmentation_configs: List of augmentation configurations
        base_seed: Base random seed for reproducibility
        cache_augmentations: Whether to pre-compute all augmentations (faster but more memory)
    
    Example:
        ```python
        from pyable_dataloader import PyableDataset, MultipleAugmentationDataset, Compose, RandomRotation
        
        # Create base dataset
        base_dataset = PyableDataset('manifest.json', target_size=[64, 64, 64])
        
        # Define augmentations
        configs = [
            {'name': 'original', 'transforms': None},
            {'name': 'rotated', 'transforms': Compose([RandomRotation([[-5,5]]*3, prob=1.0)])}
        ]
        
        # Create augmented dataset (10 subjects × 2 augmentations = 20 total samples)
        aug_dataset = MultipleAugmentationDataset(base_dataset, configs)
        
        # Use with DataLoader
        loader = DataLoader(aug_dataset, batch_size=4, shuffle=True)
        for batch in loader:
            # batch contains 4 augmented samples from different subjects/augmentations
            pass
        ```
    """
    
    def __init__(
        self,
        base_dataset: PyableDataset,
        augmentation_configs: List[Dict[str, Any]],
        base_seed: int = 42,
        cache_augmentations: bool = True
    ):
        self.base_dataset = base_dataset
        self.augmentation_configs = augmentation_configs
        self.base_seed = base_seed
        self.cache_augmentations = cache_augmentations
        
        # Pre-compute all samples if caching enabled
        if cache_augmentations:
            self._precompute_samples()
        else:
            # Calculate total length without pre-computing
            self.num_subjects = len(base_dataset)
            self.num_augmentations = len(augmentation_configs)
            self._samples = None
    
    def _precompute_samples(self):
        """Pre-compute all augmented samples."""
        self._samples = []
        
        for subject_idx in range(len(self.base_dataset)):
            augmented = self.base_dataset.get_multiple_augmentations(
                subject_idx=subject_idx,
                augmentation_configs=self.augmentation_configs,
                base_seed=self.base_seed
            )
            
            for aug_sample in augmented:
                # Stack images as channels (C × D × H × W)
                images_stacked = np.stack(aug_sample['images'], axis=0) if len(aug_sample['images']) > 1 else aug_sample['images'][0][np.newaxis, ...]
                
                sample = {
                    'id': f"{aug_sample['meta']['subject_id']}_{aug_sample['name']}",
                    'images': images_stacked,
                    'rois': aug_sample['rois'],
                    'labelmaps': aug_sample['labelmaps'],
                    'meta': aug_sample['meta'],
                    'augmentation_name': aug_sample['name']
                    # Note: augmentation_config excluded to avoid PyTorch collate issues
                }
                
                # Add label if present
                if 'label' in aug_sample['meta']:
                    sample['label'] = aug_sample['meta']['label']
                
                self._samples.append(sample)
    
    def __len__(self):
        if self.cache_augmentations:
            return len(self._samples)
        else:
            return len(self.base_dataset) * len(self.augmentation_configs)
    
    def __getitem__(self, idx: int):
        if self.cache_augmentations:
            return self._samples[idx]
        else:
            # Compute on-demand
            subject_idx = idx // self.num_augmentations
            aug_idx = idx % self.num_augmentations
            
            augmented = self.base_dataset.get_multiple_augmentations(
                subject_idx=subject_idx,
                augmentation_configs=[self.augmentation_configs[aug_idx]],
                base_seed=self.base_seed
            )
            
            aug_sample = augmented[0]
            
            # Stack images as channels (C × D × H × W)
            images_stacked = np.stack(aug_sample['images'], axis=0) if len(aug_sample['images']) > 1 else aug_sample['images'][0][np.newaxis, ...]
            
            sample = {
                'id': f"{aug_sample['meta']['subject_id']}_{aug_sample['name']}",
                'images': images_stacked,
                'rois': aug_sample['rois'],
                'labelmaps': aug_sample['labelmaps'],
                'meta': aug_sample['meta'],
                'augmentation_name': aug_sample['name']
            }
            
            # Add label if present
            if 'label' in aug_sample['meta']:
                sample['label'] = aug_sample['meta']['label']
            
            return sample
