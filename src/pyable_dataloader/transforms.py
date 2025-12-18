"""
Transform/Augmentation utilities for medical images

All transforms work with:
- images: np.ndarray (C × Z × Y × X) or (Z × Y × X)
- rois: List[np.ndarray] (each Z × Y × X)
- labelmaps: List[np.ndarray] (each Z × Y × X)
- meta: dict with spacing, origin, etc.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Union
from scipy import ndimage
import SimpleITK as sitk


def _normalize_label_values(vals) -> np.ndarray:
    """Normalize various `vals` inputs into a 1-D sorted numpy int array.

    Accepts scalars, lists, tuples, sets, numpy arrays, nested containers.
    Filters out non-convertible entries and returns a sorted unique int array.
    """
    if vals is None:
        return np.array([], dtype=int)
    if np.isscalar(vals):
        try:
            return np.array([int(vals)], dtype=int)
        except Exception:
            return np.array([], dtype=int)

    # Convert to object array to safely iterate nested containers
    vals_arr = np.asarray(vals, dtype=object)
    flat = []
    for el in np.ravel(vals_arr):
        if isinstance(el, (list, tuple, set, np.ndarray)):
            for sub in el:
                try:
                    flat.append(int(sub))
                except Exception:
                    continue
        else:
            try:
                flat.append(int(el))
            except Exception:
                continue

    if not flat:
        return np.array([], dtype=int)

    # Return sorted unique ints
    return np.array(sorted(set(flat)), dtype=int)

try:
    from pyable.imaginable import SITKImaginable, Roiable, LabelMapable
except Exception:
    # Defer import error to runtime when a transform is used that requires pyable
    SITKImaginable = None
    Roiable = None
    LabelMapable = None


class MedicalImageTransform:
    """Base class for medical image transforms."""
    
    def __call__(
        self,
        images: np.ndarray,
        rois: List[np.ndarray],
        labelmaps: List[np.ndarray],
        meta: dict
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Apply transform.
        
        Args:
            images: Image array (C × Z × Y × X) or (Z × Y × X)
            rois: List of ROI arrays (each Z × Y × X)
            labelmaps: List of labelmap arrays (each Z × Y × X)
            meta: Metadata dictionary
        
        Returns:
            Transformed (images, rois, labelmaps)
        """
        raise NotImplementedError


class Compose(MedicalImageTransform):
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[MedicalImageTransform]):
        self.transforms = transforms
    
    def __call__(self, images, rois, labelmaps, meta):
        for t in self.transforms:
            images, rois, labelmaps = t(images, rois, labelmaps, meta)
        return images, rois, labelmaps


class LabelMapOneHot(MedicalImageTransform):
    """
    Convert labelmap values to one-hot masks. Prefers explicit `values` passed to
    the transform or values present in `meta` under `meta_key`. Falls back to
    detecting unique values from the labelmap.

    Args:
        exclude_background: If True, exclude label value 0 from one-hot channels
        as_images: If True, append the one-hot channels to `images` (as additional channels).
                   Default False (returns one-hot masks in `labelmaps`).
        keep_original: If True, keep the original labelmap in the returned `labelmaps` list
                       (original will be the first element).
        meta_key: key to read/store mapping info in `meta` (useful to decode predictions).
        values: Optional explicit list of label values to use for all labelmaps.
    """

    def __init__(self, exclude_background: bool = True, as_images: bool = False,  meta_key: str = 'labelmap_values', values: Optional[List[int]] = None):
        self.exclude_background = exclude_background
        self.as_images = as_images
        self.meta_key = meta_key
        self.values = None if values is None else np.asarray(values)

    def __call__(self, images, rois, labelmaps, meta):
        if not labelmaps:
            return images, rois, labelmaps
        new_labelmaps = []
        mappings = []
        channel_map = []
        global_channel = 0

        for i, lm in enumerate(labelmaps):
            # Extract numpy array and SimpleITK image if available
            if hasattr(lm, 'getImageAsNumpy'):
                arr = lm.getImageAsNumpy()
                sitk_img = lm.getImage()
            else:
                arr = np.asarray(lm)
                sitk_img = None

            # Determine label values to encode (explicit -> meta -> image unique)
            if self.values is not None:
                vals = _normalize_label_values(self.values)
            elif meta is not None and isinstance(meta, dict) and self.meta_key in meta:
                mv = meta.get(self.meta_key)
                # support meta being list-of-lists or list-of-dicts
                if isinstance(mv, (list, tuple)) and len(mv) > i:
                    entry = mv[i]
                    if isinstance(entry, dict) and 'values' in entry:
                        vals = _normalize_label_values(entry['values'])
                    else:
                        vals = _normalize_label_values(entry)
                else:
                    vals = _normalize_label_values(mv)
            else:
                vals = _normalize_label_values(np.unique(arr))

            if self.exclude_background:
                vals = vals[vals != 0]

            # Record per-labelmap mapping
            mappings.append({'values': vals.tolist()})

            if vals.size == 0:
                # No channels for this labelmap
                continue

            channels = []
            for j, v in enumerate(vals):
                mask = None
                # Prefer Roiable path using SITK boolean comparison when possible
                if sitk_img is not None and Roiable is not None:
                    try:
                        sitk_mask = sitk_img == int(v)
                        # Ensure binary uint8 type and copy spatial info
                        try:
                            sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
                        except Exception:
                            pass
                        try:
                            sitk_mask.CopyInformation(sitk_img)
                        except Exception:
                            pass

                        roi_obj = Roiable(image=sitk_mask)
                        roi_arr = None
                        try:
                            roi_arr = roi_obj.getImageAsNumpy()
                        except Exception:
                            roi_arr = None

                        # If Roiable produced an empty image, fall back to SITK array
                        if roi_arr is None or (isinstance(roi_arr, np.ndarray) and roi_arr.sum() == 0):
                            sitk_arr = sitk.GetArrayFromImage(sitk_mask)
                            if sitk_arr.sum() > 0:
                                mask = sitk_arr.astype(np.uint8)
                            else:
                                mask = None
                        else:
                            mask = roi_arr.astype(np.uint8)
                    except Exception:
                        mask = None

                if mask is None:
                    mask = (arr == v).astype(np.uint8)

                channels.append(mask)

                # Append channel mapping
                channel_map.append({
                    'labelmap_index': int(i),
                    'label_value': int(v),
                    'local_channel': int(j),
                    'global_channel': int(global_channel)
                })
                global_channel += 1

            # Append each channel as separate 3D labelmap (backwards-compatible)
            for c in channels:
                # If original labelmap was an Imaginable, wrap channel into LabelMapable
                if hasattr(lm, 'getImage') and LabelMapable is not None:
                    try:
                        lm_obj = LabelMapable()
                        ref = lm.getImage()
                        lm_obj.setImageFromNumpy(c, refimage=ref)
                        new_labelmaps.append(lm_obj)
                    except Exception:
                        new_labelmaps.append(c)
                else:
                    new_labelmaps.append(c)

            # Optionally append channels to images
            if self.as_images:
                try:
                    # If images were Imaginable objects, wrap channel arrays into SITKImaginable
                    if isinstance(images, list) and images and hasattr(images[0], 'getImageAsNumpy'):
                        for c in channels:
                            try:
                                img_obj = SITKImaginable()
                                refimg = images[0].getImage()
                                img_obj.setImageFromNumpy(c, refimage=refimg)
                                images.append(img_obj)
                            except Exception:
                                images.append(c)
                    else:
                        if images.ndim == 4:
                            images = np.concatenate([images, np.stack(channels, axis=0)], axis=0)
                        elif images.ndim == 3:
                            images = np.concatenate([np.expand_dims(images, 0), np.stack(channels, axis=0)], axis=0)
                except Exception:
                    pass

        # Write mappings into meta for downstream decoding
        if meta is not None:
            meta[self.meta_key] = mappings
            meta_key_channels = f"{self.meta_key}_channels"
            meta[meta_key_channels] = channel_map

        return images, rois, new_labelmaps


class LabelMapContiguous(MedicalImageTransform):
    """
    Remap labelmap values to contiguous integer labels (1..N) while keeping 0 as background.

    Prefers explicit `values` passed to the transform or values present in `meta` under `meta_key`.

    Args:
        exclude_background: If True, do not include 0 in remapping (0 stays 0)
        keep_original: If True, keep the original labelmap in the returned list (first element)
        meta_key: key to read/store mapping original->contiguous in `meta` for decoding
        values: Optional explicit list of label values to use for remapping
    """

    def __init__(self, exclude_background: bool = True, keep_original: bool = False, meta_key: str = 'labelmap_mapping', values: Optional[List[int]] = None):
        self.exclude_background = exclude_background
        self.keep_original = keep_original
        self.meta_key = meta_key
        self.values = None if values is None else np.asarray(values)

    def __call__(self, images, rois, labelmaps, meta):
        if not labelmaps:
            return images, rois, labelmaps

        new_labelmaps = []
        mappings = []
        channel_map = []
        global_channel = 0

        for i, lm in enumerate(labelmaps):
            # Load array from Imaginable or numpy
            if hasattr(lm, 'getImageAsNumpy'):
                arr = lm.getImageAsNumpy()
            else:
                arr = lm if isinstance(lm, np.ndarray) else np.asarray(lm)

            # Determine label values to remap (constructor -> meta -> image unique)
            if self.values is not None:
                vals = _normalize_label_values(self.values)
            elif meta is not None and isinstance(meta, dict) and self.meta_key in meta:
                mv = meta.get(self.meta_key)
                if isinstance(mv, (list, tuple)) and len(mv) > i:
                    entry = mv[i]
                    if isinstance(entry, dict) and 'values' in entry:
                        vals = _normalize_label_values(entry['values'])
                    else:
                        vals = _normalize_label_values(entry)
                else:
                    vals = _normalize_label_values(mv)
            else:
                vals = _normalize_label_values(np.unique(arr))

            if self.exclude_background:
                vals = vals[vals != 0]

            # Build contiguous mapping: original value -> new label (1..N), 0 reserved for background
            mapping = {}
            out_dtype = arr.dtype if np.issubdtype(arr.dtype, np.integer) else np.int32
            out = np.zeros_like(arr, dtype=out_dtype)

            if vals.size > 0:
                for j, v in enumerate(vals, start=1):
                    out[arr == v] = j
                    mapping[int(v)] = int(j)

            # Optionally keep original labelmap
            if self.keep_original:
                if hasattr(lm, 'getImage') and LabelMapable is not None:
                    try:
                        orig = LabelMapable()
                        ref = lm.getImage() if hasattr(lm, 'getImage') else None
                        orig.setImageFromNumpy(arr, refimage=ref)
                        new_labelmaps.append(orig)
                    except Exception:
                        new_labelmaps.append(arr)
                else:
                    new_labelmaps.append(arr)
            # Wrap contiguous output as LabelMapable when possible
            if hasattr(lm, 'getImage') and LabelMapable is not None:
                try:
                    out_obj = LabelMapable()
                    ref = lm.getImage() if hasattr(lm, 'getImage') else None
                    out_obj.setImageFromNumpy(out, refimage=ref)
                    new_labelmaps.append(out_obj)
                except Exception:
                    new_labelmaps.append(out)
            else:
                new_labelmaps.append(out)
            mappings.append(mapping)

            # For contiguous output, each labelmap produces one output channel (the remapped map)
            channel_map.append({
                'labelmap_index': int(i),
                'label_values': vals.tolist(),
                'contiguous_map': mapping,
                'global_channel': int(global_channel)
            })
            global_channel += 1

        # Persist mapping info into meta for downstream decoding
        if meta is not None:
            meta[self.meta_key] = mappings
            meta_key_channels = f"{self.meta_key}_channels"
            meta[meta_key_channels] = channel_map

        return images, rois, new_labelmaps


class IntensityNormalization(MedicalImageTransform):
    """
    Intensity normalization.
    
    Args:
        method: 'zscore', 'minmax', or 'percentile'
        per_channel: If True, normalize each channel independently
        clip_percentile: For percentile method, clip at these percentiles
    """
    
    def __init__(
        self,
        method: str = 'zscore',
        per_channel: bool = True,
        clip_percentile: Tuple[float, float] = (1, 99)
    ):
        self.method = method
        self.per_channel = per_channel
        self.clip_percentile = clip_percentile
    
    def __call__(self, images, rois, labelmaps, meta):
        # Handle Imaginable objects (pyable) by operating on their numpy arrays
        if isinstance(images, list) and images and hasattr(images[0], 'getImageAsNumpy'):
            out = []
            for img in images:
                # Skip label/roi-like objects: do not apply intensity normalization
                if (isinstance(img, LabelMapable)) or (isinstance(img, Roiable)):
                    out.append(img)
                    continue

                arr = img.getImageAsNumpy()
                normalized = self._normalize_single_image(arr)
                # write back into the same Imaginable preserving spatial info
                try:
                    img.setImageFromNumpy(normalized, refimage=img.getImage())
                    out.append(img)
                except Exception:
                    # fallback: create a SITKImaginable container
                    obj = SITKImaginable()
                    obj.setImageFromNumpy(normalized, refimage=img.getImage())
                    out.append(obj)
            images = out
            return images, rois, labelmaps

        if hasattr(images, 'getImageAsNumpy'):
            # If this is a LabelMapable or Roiable, skip intensity normalization
            if isinstance(images, LabelMapable) or isinstance(images, Roiable):
                return images, rois, labelmaps

            arr = images.getImageAsNumpy()
            normalized = self._normalize_single_image(arr)
            images.setImageFromNumpy(normalized, refimage=images.getImage())
            return images, rois, labelmaps

        # Handle both single numpy array and list of numpy arrays
        if isinstance(images, list):
            # List of arrays (each Z × Y × X)
            normalized_images = []
            for img in images:
                normalized = self._normalize_single_image(img)
                normalized_images.append(normalized)
            images = normalized_images
        else:
            # Single stacked array (C × Z × Y × X or Z × Y × X)
            images = self._normalize_single_image(images)

        return images, rois, labelmaps
    
    def _normalize_single_image(self, images):
        if images.ndim == 3:
            # Single channel, add channel dim
            images = images[np.newaxis, ...]
            was_3d = True
        else:
            was_3d = False
        
        normalized = []
        for c in range(images.shape[0]):
            channel = images[c].copy()
            
            # Get mask (non-zero values)
            mask = channel > 0
            if not np.any(mask):
                normalized.append(channel)
                continue
            
            if self.method == 'zscore':
                mean = channel[mask].mean()
                std = channel[mask].std()
                channel = (channel - mean) / (std + 1e-8)
            
            elif self.method == 'minmax':
                min_val = channel[mask].min()
                max_val = channel[mask].max()
                channel = (channel - min_val) / (max_val - min_val + 1e-8)
            
            elif self.method == 'percentile':
                low, high = np.percentile(channel[mask], self.clip_percentile)
                channel = np.clip(channel, low, high)
                channel = (channel - low) / (high - low + 1e-8)
            
            normalized.append(channel)
        
        images = np.stack(normalized, axis=0)
        
        if was_3d:
            images = images[0]
        
        return images


class RandomFlip(MedicalImageTransform):
    """
    Random axis flip. Always applied if axes are specified.
    Args:
        axes: List of axes to flip (0=Z/D, 1=Y/H, 2=X/W)
    """
    def __init__(self, axes: List[int] = [0, 1, 2]):
        self.axes = axes

    def __call__(self, images, rois, labelmaps, meta):
        # Support both numpy arrays and pyable Imaginable objects
        for axis in self.axes:
            # Imaginable list path
            if isinstance(images, list) and images and hasattr(images[0], 'getImageAsNumpy'):
                new_images = []
                for img in images:
                    arr = img.getImageAsNumpy()
                    flipped = np.flip(arr, axis=axis)
                    # write back preserving spatial metadata
                    try:
                        img.setImageFromNumpy(flipped, refimage=img.getImage())
                        new_images.append(img)
                    except Exception:
                        # fallback: create new SITKImaginable
                        obj = SITKImaginable()
                        obj.setImageFromNumpy(flipped, refimage=img.getImage())
                        new_images.append(obj)
                images = new_images

                # rois/labelmaps are likely Roiable/LabelMapable
                new_rois = []
                for roi in rois:
                    if hasattr(roi, 'getImageAsNumpy'):
                        rarr = roi.getImageAsNumpy()
                        rfl = np.flip(rarr, axis=axis)
                        try:
                            roi.setImageFromNumpy(rfl, refimage=roi.getImage())
                            new_rois.append(roi)
                        except Exception:
                            robj = Roiable()
                            robj.setImageFromNumpy(rfl, refimage=roi.getImage())
                            new_rois.append(robj)
                    else:
                        new_rois.append(np.ascontiguousarray(np.flip(np.asarray(roi), axis=axis)))
                rois = new_rois

                new_labelmaps = []
                for lm in labelmaps:
                    if hasattr(lm, 'getImageAsNumpy'):
                        larr = lm.getImageAsNumpy()
                        lfl = np.flip(larr, axis=axis)
                        try:
                            lm.setImageFromNumpy(lfl, refimage=lm.getImage())
                            new_labelmaps.append(lm)
                        except Exception:
                            lobj = LabelMapable()
                            lobj.setImageFromNumpy(lfl, refimage=lm.getImage())
                            new_labelmaps.append(lobj)
                    else:
                        new_labelmaps.append(np.ascontiguousarray(np.flip(np.asarray(lm), axis=axis)))
                labelmaps = new_labelmaps

            else:
                # numpy array path
                if isinstance(images, list):
                    images = [np.ascontiguousarray(np.flip(img, axis=axis)) for img in images]
                else:
                    if images.ndim == 4:
                        images = np.ascontiguousarray(np.flip(images, axis=axis + 1))
                    else:
                        images = np.ascontiguousarray(np.flip(images, axis=axis))

                rois = [np.ascontiguousarray(np.flip(np.asarray(roi), axis=axis)) for roi in rois]
                labelmaps = [np.ascontiguousarray(np.flip(np.asarray(lm), axis=axis)) for lm in labelmaps]

        return images, rois, labelmaps


class RandomRotation90(MedicalImageTransform):
    """
    Random 90-degree rotation in axial plane (Y-X). Always applied.
    """
    def __init__(self):
        pass
    def __call__(self, images, rois, labelmaps, meta):
        k = np.random.randint(1, 4)  # 1, 2, or 3 (90°, 180°, 270°)
        # Rotate images
        if images.ndim == 4:
            images = np.rot90(images, k=k, axes=(-2, -1))
        else:
            images = np.rot90(images, k=k, axes=(-2, -1))
        rois = [np.rot90(roi, k=k, axes=(-2, -1)) for roi in rois]
        labelmaps = [np.rot90(lm, k=k, axes=(-2, -1)) for lm in labelmaps]
        images = np.ascontiguousarray(images)
        rois = [np.ascontiguousarray(roi) for roi in rois]
        labelmaps = [np.ascontiguousarray(lm) for lm in labelmaps]
        return images, rois, labelmaps


class RandomAffine(MedicalImageTransform):
    """
    Random affine transformation.
    
    Args:
        rotation_range: Max rotation in degrees (scalar or tuple per axis)
        zoom_range: Zoom factor range as (min, max)
        shift_range: Max shift in voxels (scalar or tuple per axis)
    """
    def __init__(
        self,
        rotation_range: Optional[Union[float, List[Tuple[float, float]]]] = 5.0,
        scale_range: Optional[Union[float, List[Tuple[float, float]]]] = 0.0,
        zoom_range: Optional[Union[float, List[Tuple[float, float]]]] = None,
        shear_range: Optional[Union[float, List[Tuple[float, float]]]] = 0.0,
        translation_range: Optional[Union[float, List[Tuple[float, float]]]] = 2.0,
        shift_range: Optional[Union[float, List[Tuple[float, float]]]] = None,
        center: Optional[Union[str, Tuple[float, float, float]]] = 'image',
        random_state: Optional[int] = None
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        # legacy/alternative parameter name for zoom
        if zoom_range is not None:
            self.zoom_range = zoom_range
            self.scale_range = zoom_range
        else:
            self.zoom_range = self.scale_range
        self.shear_range = shear_range
        self.translation_range = translation_range
        # legacy/alternative name for translation / shift
        if shift_range is not None:
            self.shift_range = shift_range
            self.translation_range = shift_range
        else:
            self.shift_range = self.translation_range
        self.center = center
        self.random_state = random_state
        # Keep init lightweight; actual affine matrix is computed per-sample in __call__
        return

    def __call__(self, images, rois, labelmaps, meta):
        # If pyable not available, skip transform
        if SITKImaginable is None:
            return images, rois, labelmaps

        # RNG
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random

        # Build reference if possible
        ref = None
        if meta and 'spacing' in meta and 'size' in meta and 'origin' in meta and 'direction' in meta:
            imgref = sitk.Image(meta['size'], sitk.sitkFloat32)
            imgref.SetSpacing(meta['spacing'])
            imgref.SetOrigin(meta['origin'])
            imgref.SetDirection(meta['direction'])
            ref = SITKImaginable(image=imgref)

        # Sample rotation (degrees) per axis
        if isinstance(self.rotation_range, (int, float)):
            rotation = [rng.uniform(-self.rotation_range, self.rotation_range) for _ in range(3)]
        else:
            rotation = [rng.uniform(low, high) for (low, high) in self.rotation_range]

        # Sample scale factors
        if isinstance(self.scale_range, (int, float)) or (isinstance(self.scale_range, (list, tuple)) and len(self.scale_range) == 2 and not isinstance(self.scale_range[0], (list, tuple))):
            smin, smax = (self.scale_range, self.scale_range) if isinstance(self.scale_range, (int, float)) else self.scale_range
            scale = [rng.uniform(smin, smax) if isinstance(smin, (int, float)) else 1.0 for _ in range(3)]
        else:
            # If scale_range is per-axis pairs
            scale = [rng.uniform(low, high) for (low, high) in self.scale_range] if isinstance(self.scale_range, (list, tuple)) else [1.0, 1.0, 1.0]

        # Sample translation in mm
        if isinstance(self.translation_range[0], (list, tuple)):
            translation = [rng.uniform(low, high) for (low, high) in self.translation_range]
        else:
            translation = [rng.uniform(-self.translation_range, self.translation_range) for _ in range(3)]

        # Convert degrees to radians
        rx, ry, rz = np.deg2rad(rotation)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        # Rotation matrices
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        rotation_matrix = Rz @ Ry @ Rx

        # Apply scale
        scale_matrix = np.diag(scale)
        affine_matrix = rotation_matrix @ scale_matrix

        # Compute offset to keep center fixed
        center_phys = None
        if meta and 'center' in meta:
            center_phys = np.asarray(meta['center'], dtype=float)
        else:
            center_phys = np.array([0.0, 0.0, 0.0])

        offset = center_phys - (affine_matrix @ center_phys) + np.asarray(translation, dtype=float)

        # Build SimpleITK affine transform
        affine = sitk.AffineTransform(3)
        affine.SetMatrix(affine_matrix.flatten().tolist())
        affine.SetTranslation(tuple(offset.tolist()))

        # Apply transform to images
        if isinstance(images, list) and images and hasattr(images[0], 'getImageAsNumpy'):
            transformed_images = []
            for img in images:
                img_copy = SITKImaginable(image=sitk.Image(img.getImage()))
                img_copy.applyTransform(affine, target_image=ref.getImage() if ref else None, interpolator=sitk.sitkLinear)
                transformed_images.append(img_copy)
            images = transformed_images
        else:
            # numpy path
            if isinstance(images, list):
                transformed_images = []
                for img in images:
                    img_obj = SITKImaginable()
                    img_obj.setImageFromNumpy(np.asarray(img), refimage=ref.getImage() if ref else None)
                    img_obj.applyTransform(affine, target_image=ref.getImage() if ref else None, interpolator=sitk.sitkLinear)
                    transformed_images.append(img_obj.getImageAsNumpy())
                images = transformed_images
            else:
                if images.ndim == 4:
                    channels = []
                    for c in range(images.shape[0]):
                        img_obj = SITKImaginable()
                        img_obj.setImageFromNumpy(np.asarray(images[c]), refimage=ref.getImage() if ref else None)
                        img_obj.applyTransform(affine, target_image=ref.getImage() if ref else None, interpolator=sitk.sitkLinear)
                        channels.append(img_obj.getImageAsNumpy())
                    images = np.stack(channels, axis=0)
                else:
                    img_obj = SITKImaginable()
                    img_obj.setImageFromNumpy(np.asarray(images), refimage=ref.getImage() if ref else None)
                    img_obj.applyTransform(affine, target_image=ref.getImage() if ref else None, interpolator=sitk.sitkLinear)
                    images = img_obj.getImageAsNumpy()

        # Labelmaps and rois (nearest neighbor)
        if isinstance(labelmaps, list) and labelmaps and hasattr(labelmaps[0], 'getImageAsNumpy'):
            transformed_labelmaps = []
            for lm in labelmaps:
                lm_obj = LabelMapable(image=sitk.Image(lm.getImage()))
                lm_obj.applyTransform(affine, target_image=ref.getImage() if ref else None)
                transformed_labelmaps.append(lm_obj)
            labelmaps = transformed_labelmaps
        else:
            transformed_labelmaps = []
            for lm in labelmaps:
                l = LabelMapable()
                l.setImageFromNumpy(np.asarray(lm), refimage=ref.getImage() if ref else None)
                l.applyTransform(affine, target_image=ref.getImage() if ref else None)
                transformed_labelmaps.append(l.getImageAsNumpy())
            labelmaps = transformed_labelmaps

        if isinstance(rois, list) and rois and hasattr(rois[0], 'getImageAsNumpy'):
            transformed_rois = []
            for roi in rois:
                roi_obj = Roiable(image=sitk.Image(roi.getImage()))
                roi_obj.applyTransform(affine, target_image=ref.getImage() if ref else None)
                transformed_rois.append(roi_obj)
            rois = transformed_rois
        else:
            transformed_rois = []
            for roi in rois:
                r = LabelMapable()
                r.setImageFromNumpy(np.asarray(roi), refimage=ref.getImage() if ref else None)
                r.applyTransform(affine, target_image=ref.getImage() if ref else None)
                transformed_rois.append(r.getImageAsNumpy())
            rois = transformed_rois

        return images, rois, labelmaps


class RandomTranslation(MedicalImageTransform):
    """
    Random translation using Imaginable to preserve labels.

    Args:
        translation_range: per-axis list of (min, max) in mm (len=3)
    """
    def __init__(self, translation_range: Optional[List[Tuple[float, float]]] = None, center: Optional[Union[str, Tuple[float, float, float]]] = 'image', random_state: Optional[int] = None):
        # Default ±5 mm per axis
        if translation_range is None:
            translation_range = [[-5, 5], [-5, 5], [-5, 5]]
        self.translation_range = translation_range
        self.center = center
        self.random_state = random_state

    def __call__(self, images, rois, labelmaps, meta):

        if SITKImaginable is None:
            # Can't apply label-preserving transform, skip
            return images, rois, labelmaps

        # Random generator for reproducibility
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random

        # Sample per-axis translations in mm
        t = [rng.uniform(low, high) for (low, high) in self.translation_range]

        # Check if images is list of pyable objects or numpy arrays
        if isinstance(images, list) and images and hasattr(images[0], 'getImage'):
            # Apply to pyable objects
            transformed_images = []
            for img in images:
                img_copy = SITKImaginable(image=sitk.Image(img.getImage()))
                img_copy.translateImage(t, interpolator=sitk.sitkLinear)
                transformed_images.append(img_copy)
            images = transformed_images

            transformed_rois = []
            for roi in rois:
                roi_copy = Roiable(image=sitk.Image(roi.getImage()))
                roi_copy.translateImage(t, interpolator=sitk.sitkNearestNeighbor)
                transformed_rois.append(roi_copy)
            rois = transformed_rois

            transformed_labelmaps = []
            for lm in labelmaps:
                lm_copy = LabelMapable(image=sitk.Image(lm.getImage()))
                lm_copy.translateImage(t, interpolator=sitk.sitkNearestNeighbor)
                transformed_labelmaps.append(lm_copy)
            labelmaps = transformed_labelmaps
        else:
            # Handle numpy arrays (single array or list of arrays)
            # Create reference
            ref = None
            if meta and 'spacing' in meta and 'size' in meta and 'origin' in meta and 'direction' in meta:
                imgref = sitk.Image(meta['size'], sitk.sitkFloat32)
                imgref.SetSpacing(meta['spacing'])
                imgref.SetOrigin(meta['origin'])
                imgref.SetDirection(meta['direction'])
                ref = SITKImaginable(image=imgref)

            if isinstance(images, list):
                # List of arrays (each Z × Y × X)
                transformed_images = []
                for img in images:
                    img_copy = SITKImaginable()
                    img_copy.setImageFromNumpy(img, refimage=ref.getImage() if ref else None)
                    img_copy.translateImage(t, interpolator=sitk.sitkLinear, reference_image=ref.getImage() if ref else None)
                    transformed_images.append(img_copy.getImageAsNumpy())
                images = transformed_images
            else:
                # Single stacked array
                transformed_images = []
                if images.ndim == 4:
                    for c in range(images.shape[0]):
                        img = SITKImaginable()
                        img.setImageFromNumpy(images[c], refimage=ref.getImage() if ref else None)
                        img.translateImage(t, interpolator=sitk.sitkLinear, reference_image=ref.getImage() if ref else None)
                        transformed_images.append(img.getImageAsNumpy())
                    images = np.stack(transformed_images, axis=0)
                else:
                    img = SITKImaginable()
                    img.setImageFromNumpy(images, refimage=ref.getImage() if ref else None)
                    img.translateImage(t, interpolator=sitk.sitkLinear, reference_image=ref.getImage() if ref else None)
                    images = img.getImageAsNumpy()

            # Transform ROIs and labelmaps
            transformed_rois = []
            for roi in rois:
                l = LabelMapable()
                l.setImageFromNumpy(roi, refimage=ref.getImage() if ref else None)
                l.translateImage(t, interpolator=sitk.sitkNearestNeighbor, reference_image=ref.getImage() if ref else None)
                transformed_rois.append(l.getImageAsNumpy())
            rois = transformed_rois

            transformed_labelmaps = []
            for lm in labelmaps:
                l = LabelMapable()
                l.setImageFromNumpy(lm, refimage=ref.getImage() if ref else None)
                l.translateImage(t, interpolator=sitk.sitkNearestNeighbor, reference_image=ref.getImage() if ref else None)
                transformed_labelmaps.append(l.getImageAsNumpy())
            labelmaps = transformed_labelmaps

        return images, rois, labelmaps


class RandomRotation(MedicalImageTransform):
    """
    Random rotation using Imaginable, with per-axis ranges in degrees.
    """

    def __init__(self, rotation_range: Optional[List[Tuple[float, float]]] = None, center: Optional[Union[str, Tuple[float, float, float]]] = 'image', random_state: Optional[int] = None):
        if rotation_range is None:
            rotation_range = [[-5, 5], [-5, 5], [-5, 5]]
        self.rotation_range = rotation_range
        self.center = center
        self.random_state = random_state

    def __call__(self, images, rois, labelmaps, meta):

        if SITKImaginable is None:
            return images, rois, labelmaps

        # Create reference
        ref = None
        if meta and 'spacing' in meta and 'size' in meta and 'origin' in meta and 'direction' in meta:
            imgref = sitk.Image(meta['size'], sitk.sitkFloat32)
            imgref.SetSpacing(meta['spacing'])
            imgref.SetOrigin(meta['origin'])
            imgref.SetDirection(meta['direction'])
            ref = SITKImaginable(image=imgref)

        # RNG for reproducibility
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random

        # Sample rotations per axis
        rotation = [rng.uniform(low, high) for (low, high) in self.rotation_range]

        if isinstance(images, list) and images and hasattr(images[0], 'getImage'):
            # List of Imaginable objects
            transformed_images = []
            for img in images:
                img_copy = SITKImaginable(image=sitk.Image(img.getImage()))
                img_copy.rotateImage(rotation, interpolator=sitk.sitkLinear, reference_image=ref.getImage() if ref else None)
                transformed_images.append(img_copy)
            images = transformed_images
        else:
            # Single stacked array
            transformed_images = []
            if images.ndim == 4:
                for c in range(images.shape[0]):
                    img = SITKImaginable()
                    img.setImageFromNumpy(images[c], refimage=ref.getImage() if ref else None)
                    img.rotateImage(rotation, interpolator=sitk.sitkLinear, reference_image=ref.getImage() if ref else None)
                    transformed_images.append(img.getImageAsNumpy())
                images = np.stack(transformed_images, axis=0)
            else:
                img = SITKImaginable()
                img.setImageFromNumpy(images, refimage=ref.getImage() if ref else None)
                img.rotateImage(rotation, interpolator=sitk.sitkLinear, reference_image=ref.getImage() if ref else None)
                images = img.getImageAsNumpy()

        # If images were Imaginable objects, ensure labelmaps/rois are returned as Imaginable objects too
        if isinstance(images, list) and images and hasattr(images[0], 'getImage'):
            transformed_rois = []
            for roi in rois:
                roi_copy = Roiable(image=sitk.Image(roi.getImage()))
                roi_copy.rotateImage(rotation, interpolator=sitk.sitkNearestNeighbor, reference_image=ref.getImage() if ref else None)
                transformed_rois.append(roi_copy)
            rois = transformed_rois

            transformed_labelmaps = []
            for lm in labelmaps:
                lm_copy = LabelMapable(image=sitk.Image(lm.getImage()))
                lm_copy.rotateImage(rotation, interpolator=sitk.sitkNearestNeighbor, reference_image=ref.getImage() if ref else None)
                transformed_labelmaps.append(lm_copy)
            labelmaps = transformed_labelmaps
        else:
            transformed_rois = []
            for roi in rois:
                l = LabelMapable()
                l.setImageFromNumpy(roi, refimage=ref.getImage() if ref else None)
                l.rotateImage(rotation, interpolator=sitk.sitkNearestNeighbor, reference_image=ref.getImage() if ref else None)
                transformed_rois.append(l.getImageAsNumpy())
            rois = transformed_rois

            transformed_labelmaps = []
            for lm in labelmaps:
                l = LabelMapable()
                l.setImageFromNumpy(lm, refimage=ref.getImage() if ref else None)
                l.rotateImage(rotation, interpolator=sitk.sitkNearestNeighbor, reference_image=ref.getImage() if ref else None)
                transformed_labelmaps.append(l.getImageAsNumpy())
            labelmaps = transformed_labelmaps

        return images, rois, labelmaps


class RandomBSpline(MedicalImageTransform):
    """
    Random BSpline transform using SimpleITK BSplineTransform initializer.

    Args:
        mesh_size: control point mesh size (tuple of 3 ints)
        magnitude: maximum perturbation in mm
    """
    def __init__(self, mesh_size=(4, 4, 4), magnitude: float = 5.0):
        self.mesh_size = mesh_size
        self.magnitude = magnitude

    def __call__(self, images, rois, labelmaps, meta):

        if SITKImaginable is None:
            return images, rois, labelmaps

        # Create reference
        if not (meta and 'spacing' in meta and 'size' in meta and 'origin' in meta and 'direction' in meta):
            return images, rois, labelmaps

        imgref = sitk.Image(meta['size'], sitk.sitkFloat32)
        imgref.SetSpacing(meta['spacing'])
        imgref.SetOrigin(meta['origin'])
        imgref.SetDirection(meta['direction'])
        ref = SITKImaginable(image=imgref)

        # Initialize BSpline transform
        bs = sitk.BSplineTransformInitializer(imgref, self.mesh_size)
        params = list(bs.GetParameters())
        # Randomly perturb parameters (every 3rd parameter corresponds to x,y,z offsets)
        for i in range(len(params)):
            params[i] += np.random.uniform(-self.magnitude, self.magnitude)
        bs.SetParameters(params)

        # Apply to images and labelmaps
        if isinstance(images, list):
            # List of arrays (each Z × Y × X)
            transformed_images = []
            for img in images:
                img_copy = SITKImaginable()
                img_copy.setImageFromNumpy(img, refimage=ref.getImage())
                img_copy.applyTransform(bs, target_image=ref.getImage(), interpolator='linear')
                transformed_images.append(img_copy.getImageAsNumpy())
            images = transformed_images
        else:
            # Single stacked array
            transformed_images = []
            if images.ndim == 4:
                for c in range(images.shape[0]):
                    img = SITKImaginable()
                    img.setImageFromNumpy(images[c], refimage=ref.getImage())
                    img.applyTransform(bs, target_image=ref.getImage(), interpolator='linear')
                    transformed_images.append(img.getImageAsNumpy())
                images = np.stack(transformed_images, axis=0)
            else:
                img = SITKImaginable()
                img.setImageFromNumpy(images, refimage=ref.getImage())
                img.applyTransform(bs, target_image=ref.getImage(), interpolator='linear')
                images = img.getImageAsNumpy()

        transformed_rois = []
        for roi in rois:
            l = LabelMapable()
            l.setImageFromNumpy(roi, refimage=ref.getImage())
            l.applyTransform(bs, target_image=ref.getImage())
            transformed_rois.append(l.getImageAsNumpy())
        rois = transformed_rois

        transformed_labelmaps = []
        for lm in labelmaps:
            l = LabelMapable()
            l.setImageFromNumpy(lm, refimage=ref.getImage())
            l.applyTransform(bs, target_image=ref.getImage())
            transformed_labelmaps.append(l.getImageAsNumpy())
        labelmaps = transformed_labelmaps

        return images, rois, labelmaps


class RandomNoise(MedicalImageTransform):
    """
    Add random Gaussian noise.
    
    Args:
        std: Standard deviation of noise (relative to image values)
        prob: Probability of adding noise
    """
    
    def __init__(self, std: float = 0.01, prob: float = 0.5):
        self.std = std
        self.prob = prob
    
    def __call__(self, images, rois, labelmaps, meta):
        # Only apply noise to image data. Leave labelmaps and rois unchanged.
        if np.random.rand() < self.prob:
            # Imaginable objects path: list of Imaginable or single Imaginable
            if isinstance(images, list) and images and hasattr(images[0], 'getImageAsNumpy'):
                new_images = []
                for img in images:
                    # Skip label/roi-like objects (shouldn't appear in images, but guard anyway)
                    if (isinstance(img, LabelMapable)) or (isinstance(img, Roiable)):
                        new_images.append(img)
                        continue

                    arr = img.getImageAsNumpy()
                    # Generate noise with same shape
                    noise = np.random.normal(0, self.std, arr.shape)
                    mask = arr > 0
                    arr_noisy = arr + noise * mask
                    arr_noisy = np.clip(arr_noisy, 0, None)
                    try:
                        img.setImageFromNumpy(arr_noisy, refimage=img.getImage())
                        new_images.append(img)
                    except Exception:
                        obj = SITKImaginable()
                        obj.setImageFromNumpy(arr_noisy, refimage=img.getImage())
                        new_images.append(obj)
                images = new_images

            elif hasattr(images, 'getImageAsNumpy'):
                # Single Imaginable
                if not (isinstance(images, LabelMapable) or isinstance(images, Roiable)):
                    arr = images.getImageAsNumpy()
                    noise = np.random.normal(0, self.std, arr.shape)
                    mask = arr > 0
                    arr_noisy = arr + noise * mask
                    arr_noisy = np.clip(arr_noisy, 0, None)
                    images.setImageFromNumpy(arr_noisy, refimage=images.getImage())

            else:
                # Numpy path (stacked or list)
                if isinstance(images, list):
                    new_images = []
                    for img in images:
                        arr = np.asarray(img)
                        noise = np.random.normal(0, self.std, arr.shape)
                        mask = arr > 0
                        arr_noisy = arr + noise * mask
                        arr_noisy = np.clip(arr_noisy, 0, None)
                        new_images.append(np.ascontiguousarray(arr_noisy))
                    images = new_images
                else:
                    arr = np.asarray(images)
                    noise = np.random.normal(0, self.std, arr.shape)
                    mask = arr > 0
                    arr_noisy = arr + noise * mask
                    arr_noisy = np.clip(arr_noisy, 0, None)
                    images = np.ascontiguousarray(arr_noisy)

        return images, rois, labelmaps


class CropOrPad(MedicalImageTransform):
    """
    Crop or pad to target size.
    
    Args:
        target_size: Target size [D, H, W]
        mode: 'center' or 'random'
    """
    
    def __init__(self, target_size: List[int], mode: str = 'center'):
        self.target_size = target_size
        self.mode = mode
    
    def _get_crop_pad_coords(self, current_size, target_size):
        """Calculate crop/pad coordinates."""
        coords = []
        for curr, tgt in zip(current_size, target_size):
            if curr > tgt:
                # Need to crop
                if self.mode == 'center':
                    start = (curr - tgt) // 2
                else:  # random
                    start = np.random.randint(0, curr - tgt + 1)
                coords.append((start, start + tgt))
            else:
                # No crop needed
                coords.append((0, curr))
        return coords
    
    def _get_pad_widths(self, current_size, target_size):
        """Calculate padding widths."""
        widths = []
        for curr, tgt in zip(current_size, target_size):
            if curr < tgt:
                diff = tgt - curr
                if self.mode == 'center':
                    pad_before = diff // 2
                    pad_after = diff - pad_before
                else:  # random
                    pad_before = np.random.randint(0, diff + 1)
                    pad_after = diff - pad_before
                widths.append((pad_before, pad_after))
            else:
                widths.append((0, 0))
        return widths
    
    def __call__(self, images, rois, labelmaps, meta):
        if images.ndim == 4:
            _, D, H, W = images.shape
            current_size = [D, H, W]
        else:
            D, H, W = images.shape
            current_size = [D, H, W]
        
        # Crop if needed
        crop_coords = self._get_crop_pad_coords(current_size, self.target_size)
        
        if images.ndim == 4:
            images = images[
                :,
                crop_coords[0][0]:crop_coords[0][1],
                crop_coords[1][0]:crop_coords[1][1],
                crop_coords[2][0]:crop_coords[2][1]
            ]
        else:
            images = images[
                crop_coords[0][0]:crop_coords[0][1],
                crop_coords[1][0]:crop_coords[1][1],
                crop_coords[2][0]:crop_coords[2][1]
            ]
        # Pad if needed
        pad_widths = self._get_pad_widths(current_size, self.target_size)
        
        if any(w != (0, 0) for w in pad_widths):
            if images.ndim == 4:
                pad_widths_full = [(0, 0)] + pad_widths
            else:
                pad_widths_full = pad_widths
            
            images = np.pad(images, pad_widths_full, mode='constant', constant_values=0)
            
            rois = [
                np.pad(roi, pad_widths, mode='constant', constant_values=0)
                for roi in rois
            ]
            
            labelmaps = [
                np.pad(lm, pad_widths, mode='constant', constant_values=0)
                for lm in labelmaps
            ]
        
        return images, rois, labelmaps
