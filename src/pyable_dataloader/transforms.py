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
        # Handle both single array and list of arrays
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
    Random axis flip.
    
    Args:
        axes: List of axes to potentially flip (0=Z/D, 1=Y/H, 2=X/W)
        prob: Probability of flipping each axis
    """
    
    def __init__(self, axes: List[int] = [0, 1, 2], prob: float = 0.5):
        self.axes = axes
        self.prob = prob
    
    def __call__(self, images, rois, labelmaps, meta):
        for axis in self.axes:
            if np.random.rand() < self.prob:
                # Handle both single array and list of arrays
                if isinstance(images, list):
                    # List of arrays
                    images = [np.flip(img, axis=axis) for img in images]
                else:
                    # Single stacked array
                    if images.ndim == 4:
                        # C × Z × Y × X: axis + 1 to account for channel
                        images = np.flip(images, axis=axis + 1)
                    else:
                        # Z × Y × X
                        images = np.flip(images, axis=axis)
                
                # Flip ROIs and labelmaps
                rois = [np.flip(roi, axis=axis) for roi in rois]
                labelmaps = [np.flip(lm, axis=axis) for lm in labelmaps]
        
        # Make contiguous
        if isinstance(images, list):
            images = [np.ascontiguousarray(img) for img in images]
        else:
            images = np.ascontiguousarray(images)
        rois = [np.ascontiguousarray(roi) for roi in rois]
        labelmaps = [np.ascontiguousarray(lm) for lm in labelmaps]
        
        return images, rois, labelmaps


class RandomRotation90(MedicalImageTransform):
    """
    Random 90-degree rotation in axial plane (Y-X).
    
    Args:
        prob: Probability of rotation
    """
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, images, rois, labelmaps, meta):
        if np.random.rand() < self.prob:
            k = np.random.randint(1, 4)  # 1, 2, or 3 (90°, 180°, 270°)
            
            # Rotate images
            if images.ndim == 4:
                # C × Z × Y × X: rotate in Y-X plane (axes -2, -1)
                images = np.rot90(images, k=k, axes=(-2, -1))
            else:
                # Z × Y × X
                images = np.rot90(images, k=k, axes=(-2, -1))
            
            # Rotate ROIs and labelmaps
            rois = [np.rot90(roi, k=k, axes=(-2, -1)) for roi in rois]
            labelmaps = [np.rot90(lm, k=k, axes=(-2, -1)) for lm in labelmaps]
            
            # Make contiguous
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
        prob: Probability of applying transform
    """
    
    def __init__(
        self,
        rotation_range: Optional[Union[float, List[Tuple[float, float]]]] = 5.0,
        zoom_range: Tuple[float, float] = (0.95, 1.05),
        shift_range: Optional[Union[float, List[Tuple[float, float]]]] = 2.0,
        prob: float = 0.5
    ):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.prob = prob
    
    def __call__(self, images, rois, labelmaps, meta):
        if np.random.rand() > self.prob:
            return images, rois, labelmaps
        
        # Generate random parameters
        if isinstance(self.rotation_range, (list, tuple)):
            rotation = [np.random.uniform(low, high) for (low, high) in self.rotation_range]
        else:
            rotation = [0.0, 0.0, float(np.random.uniform(-self.rotation_range, self.rotation_range))]
        zoom = np.random.uniform(*self.zoom_range)
        if isinstance(self.shift_range, (list, tuple)):
            shift = [np.random.uniform(low, high) for (low, high) in self.shift_range]
        else:
            shift = [float(np.random.uniform(-self.shift_range, self.shift_range)) for _ in range(3)]
        
        # Build transformation matrix
        # For 3D: rotate in axial plane (Y-X)
        
        # Determine shape
        if images.ndim == 4:
            _, D, H, W = images.shape
            has_channels = True
        else:
            D, H, W = images.shape
            has_channels = False
        
        center = np.array([D, H, W]) / 2.0
        
        # Create affine matrix (rotation + zoom)
        # Convert Euler angles in degrees -> radians
        rx, ry, rz = np.deg2rad(rotation)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        rotation_matrix = Rz @ Ry @ Rx
        
        # Apply zoom
        zoom_matrix = np.diag([zoom, zoom, zoom])
        affine_matrix = rotation_matrix @ zoom_matrix
        
        # Compute offset to keep center fixed
        offset = center - affine_matrix @ center + shift
        
        # If pyable is available, prefer using SITK transforms for label-preserving
        # behavior via Imaginable.applyTransform. Otherwise fall back to ndimage.
        if SITKImaginable is not None:
            # Build an affine matrix and translation in mm using meta
            # Convert arrays into SITK Imaginable objects, apply transform, and
            # return the numpy arrays.
            # Build transform
            dimension = 3
            # Use rotation sampled above
            # Use center = image center
            # Convert images into sitk imaginables
            # Create reference from meta
            ref = None
            if meta and 'spacing' in meta and 'size' in meta and 'origin' in meta and 'direction' in meta:
                imgref = sitk.Image(meta['size'], sitk.sitkFloat32)
                imgref.SetSpacing(meta['spacing'])
                imgref.SetOrigin(meta['origin'])
                imgref.SetDirection(meta['direction'])
                ref = SITKImaginable(image=imgref)

            # Convert and apply transform per channel/element
            if isinstance(images, list):
                # List of arrays (each Z × Y × X)
                transformed_images = []
                for img in images:
                    img_copy = SITKImaginable()
                    img_copy.setImageFromNumpy(img, refimage=ref.getImage() if ref else None)
                    img_copy.rotateImage(rotation, translation=shift, interpolator=sitk.sitkLinear, reference_image=ref.getImage() if ref else None)
                    transformed_images.append(img_copy.getImageAsNumpy())
                images = transformed_images
            else:
                # Single stacked array
                transformed_images = []
                if images.ndim == 4:
                    for c in range(images.shape[0]):
                        img = SITKImaginable()
                        img.setImageFromNumpy(images[c], refimage=ref.getImage() if ref else None)
                        img.rotateImage(rotation, translation=shift, interpolator=sitk.sitkLinear, reference_image=ref.getImage() if ref else None)
                        transformed_images.append(img.getImageAsNumpy())
                else:
                    img = SITKImaginable()
                    img.setImageFromNumpy(images, refimage=ref.getImage() if ref else None)
                    img.rotateImage(rotation, translation=shift, interpolator=sitk.sitkLinear, reference_image=ref.getImage() if ref else None)
                    transformed_images = [img.getImageAsNumpy()]

                # Stack channels if needed
                if images.ndim == 4:
                    images = np.stack(transformed_images, axis=0)
                else:
                    images = transformed_images[0]

            # Apply nearest-neighbor transforms to rois and labelmaps
            transformed_rois = []
            for roi in rois:
                l = LabelMapable()
                l.setImageFromNumpy(roi, refimage=ref.getImage() if ref else None)
                l.rotateImage(rotation, translation=shift, interpolator=sitk.sitkNearestNeighbor, reference_image=ref.getImage() if ref else None)
                transformed_rois.append(l.getImageAsNumpy())

            transformed_labelmaps = []
            for lm in labelmaps:
                l = LabelMapable()
                l.setImageFromNumpy(lm, refimage=ref.getImage() if ref else None)
                l.rotateImage(rotation, translation=shift, interpolator=sitk.sitkNearestNeighbor, reference_image=ref.getImage() if ref else None)
                transformed_labelmaps.append(l.getImageAsNumpy())

            rois = transformed_rois
            labelmaps = transformed_labelmaps
            return images, rois, labelmaps

        # Otherwise fall back to the existing implementation
        # Generate random parameters
        if isinstance(self.rotation_range, (list, tuple)):
            angle = float(np.random.uniform(-self.rotation_range[2][0], self.rotation_range[2][1]))
        else:
            angle = float(np.random.uniform(-self.rotation_range, self.rotation_range))
        zoom = np.random.uniform(*self.zoom_range)
        if isinstance(self.shift_range, (list, tuple)):
            # Convert mm to voxels if meta available
            if meta and 'spacing' in meta:
                shift = np.array([np.random.uniform(low, high) / s for (low, high), s in zip(self.shift_range, meta['spacing'])])
            else:
                shift = np.array([np.random.uniform(low, high) for (low, high) in self.shift_range])
        else:
            if meta and 'spacing' in meta:
                shift = float(np.random.uniform(-self.shift_range, self.shift_range)) / np.array(meta['spacing'])
            else:
                shift = np.random.uniform(-self.shift_range, self.shift_range, size=3)
        if has_channels:
            transformed_images = []
            for c in range(images.shape[0]):
                transformed = ndimage.affine_transform(
                    images[c],
                    matrix=np.linalg.inv(affine_matrix),
                    offset=offset,
                    order=1,
                    mode='constant',
                    cval=0
                )
                transformed_images.append(transformed)
            images = np.stack(transformed_images, axis=0)
        else:
            images = ndimage.affine_transform(
                images,
                matrix=np.linalg.inv(affine_matrix),
                offset=offset,
                order=1,
                mode='constant',
                cval=0
            )
        
        # Apply to ROIs and labelmaps (use order=0 for nearest neighbor)
        rois = [
            ndimage.affine_transform(
                roi,
                matrix=np.linalg.inv(affine_matrix),
                offset=offset,
                order=0,
                mode='constant',
                cval=0
            )
            for roi in rois
        ]
        
        labelmaps = [
            ndimage.affine_transform(
                lm,
                matrix=np.linalg.inv(affine_matrix),
                offset=offset,
                order=0,
                mode='constant',
                cval=0
            )
            for lm in labelmaps
        ]
        
        return images, rois, labelmaps


class RandomTranslation(MedicalImageTransform):
    """
    Random translation using Imaginable to preserve labels.

    Args:
        translation_range: per-axis list of (min, max) in mm (len=3)
        prob: probability to apply
    """

    def __init__(self, translation_range: Optional[List[Tuple[float, float]]] = None, prob: float = 0.5):
        # Default ±5 mm per axis
        if translation_range is None:
            translation_range = [[-5, 5], [-5, 5], [-5, 5]]
        self.translation_range = translation_range
        self.prob = prob

    def __call__(self, images, rois, labelmaps, meta):
        if np.random.rand() > self.prob:
            return images, rois, labelmaps

        if SITKImaginable is None:
            # Can't apply label-preserving transform, skip
            return images, rois, labelmaps

        # Sample per-axis translations in mm
        t = [np.random.uniform(low, high) for (low, high) in self.translation_range]

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

    def __init__(self, rotation_range: Optional[List[Tuple[float, float]]] = None, prob: float = 0.5):
        if rotation_range is None:
            rotation_range = [[-5, 5], [-5, 5], [-5, 5]]
        self.rotation_range = rotation_range
        self.prob = prob

    def __call__(self, images, rois, labelmaps, meta):
        if np.random.rand() > self.prob:
            return images, rois, labelmaps

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

        # Sample rotations per axis
        rotation = [np.random.uniform(low, high) for (low, high) in self.rotation_range]

        # Sample rotations per axis
        rotation = [np.random.uniform(low, high) for (low, high) in self.rotation_range]

        if isinstance(images, list):
            # List of arrays (each Z × Y × X)
            transformed_images = []
            for img in images:
                img_copy = SITKImaginable()
                img_copy.setImageFromNumpy(img, refimage=ref.getImage() if ref else None)
                img_copy.rotateImage(rotation, interpolator=sitk.sitkLinear, reference_image=ref.getImage() if ref else None)
                transformed_images.append(img_copy.getImageAsNumpy())
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
        prob: probability to apply
    """

    def __init__(self, mesh_size=(4, 4, 4), magnitude: float = 5.0, prob: float = 0.5):
        self.mesh_size = mesh_size
        self.magnitude = magnitude
        self.prob = prob

    def __call__(self, images, rois, labelmaps, meta):
        if np.random.rand() > self.prob:
            return images, rois, labelmaps

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
        if np.random.rand() < self.prob:
            # Generate noise
            noise = np.random.normal(0, self.std, images.shape)
            
            # Apply only to foreground (non-zero) voxels
            if images.ndim == 4:
                mask = images > 0
            else:
                mask = images > 0
            
            images = images + noise * mask
            images = np.clip(images, 0, None)  # Ensure non-negative
        
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
        
        rois = [
            roi[
                crop_coords[0][0]:crop_coords[0][1],
                crop_coords[1][0]:crop_coords[1][1],
                crop_coords[2][0]:crop_coords[2][1]
            ]
            for roi in rois
        ]
        
        labelmaps = [
            lm[
                crop_coords[0][0]:crop_coords[0][1],
                crop_coords[1][0]:crop_coords[1][1],
                crop_coords[2][0]:crop_coords[2][1]
            ]
            for lm in labelmaps
        ]
        
        # Update current size
        if images.ndim == 4:
            current_size = list(images.shape[1:])
        else:
            current_size = list(images.shape)
        
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
