import random


from pyable.imaginable import create_affine_matrix
    
class Compose:
    """
    Compose several transforms together.
    Args:
        transforms (list): List of transform instances.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, ables, meta=None):
        for t in self.transforms:
            ables = t(ables, meta)
        return ables

class RandomRototranslation:
    """
    Apply a random rotation and optional translation to images (not labelmaps or rois).
    Rotates by a random angle (in degrees) within the given range for each axis.
    Optionally applies random translation within a given range.
    Args:
        angle_range (tuple or list): (min_angle, max_angle) in degrees, or list of 3 tuples for 3D.
        translation_range (tuple or list, optional): (min, max) in mm, or list of 3 tuples for 3D. Default: None (no translation).
    """
    def __init__(self, angle_range=((-10, 10), (-10, 10), (-10, 10)), translation_range=None):
        self.angle_range = angle_range
        self.translation_range = translation_range

    def __call__(self, ables, meta=None):
        
        if isinstance(self.angle_range[0], (tuple, list)):
            angles = [random.uniform(r[0], r[1]) for r in self.angle_range]
        else:
            angles = [random.uniform(self.angle_range[0], self.angle_range[1]) for _ in range(a.getImageDimension())]
        # Support per-axis or single range for translation
        translation = None
        if self.translation_range is not None:
            if isinstance(self.translation_range[0], (tuple, list)):
                translation = [random.uniform(r[0], r[1]) for r in self.translation_range]
            else:
                translation = [random.uniform(self.translation_range[0], self.translation_range[1]) for _ in range(a.getImageDimension())]
                
        for i, a in enumerate(ables):
            # Determine center: from meta if available, else from image
            center = None
            if meta and 'center' in meta:
                center = meta['center']
            else:
                center = a.getImageCenterCoordinate()
            # Support per-axis or single range for rotation
        
            # Pad image so all corners remain after transform
            safe_a = SafeImaginable(image=a.getImage())
            safe_a.safepaddingfortransform(angles, center=center, translation=translation)
            # Resample original able to padded target, preserving class
            a.resampleOnTargetImage(safe_a)
            a.rotateImage(angles, center=center, translation=translation)
        return ables

# =============================
# RandomAffineTransform
# =============================
def xyz_to_numpy_axis(axis):
    # axis: 0 (x), 1 (y), 2 (z)
    # returns: numpy axis for (z, y, x) ordering
    mapping = {0: 2, 1: 1, 2: 0}
    return mapping[axis]

# Example: flip along x in xyz
class FlipDimensions:
    """
    Randomly flip images (not labelmaps or rois) along specified dimensions with given probability.
    Args:
        axis (list): List of axis indices to consider for flipping (e.g., [0, 1, 2] for 3D).
        p (float): Probability of flipping along each axis.
    """
    def __init__(self, axis=[0, 1, 2]):
        self.axis = axis
    
    def __call__(self, ables, meta=None):
        # Support both numpy arrays and pyable Imaginable objects
        for axis in self.axis:
            # Imaginable list path
            for ab in ables:
                arr = ab.getImageAsNumpy()                
                flipped = np.flip(arr, axis=xyz_to_numpy_axis(axis)) # for numpy zyx ordering
                # write back preserving spatial metadata
                ab.setImageFromNumpy(flipped)
        return ables
    
class RandomAffineTransform:
    """
    Apply a random affine transformation (scale, rotation, translation) to images (not labelmaps or rois).
    Args:
        scale_range (tuple or list): (min_scale, max_scale) or list of 3 tuples for 3D.
        angle_range (tuple or list): (min_angle, max_angle) in degrees, or list of 3 tuples for 3D.
        translation_range (tuple or list): (min, max) in mm, or list of 3 tuples for 3D.
    """
    def __init__(self, scale_range=(0.95, 1.05), angle_range=((-10, 10), (-10, 10), (-10, 10)), translation_range=((-5, 5), (-5, 5), (-5, 5))):
        self.scale_range = scale_range
        self.angle_range = angle_range
        self.translation_range = translation_range

    def __call__(self, ables, meta=None):

        center = None
        if meta and 'center' in meta:
            center = meta['center']
        else:
            center = a.getImageCenterCoordinate()
        # Support per-axis or single range for rotation
        if isinstance(self.angle_range[0], (tuple, list)):
            angles = [random.uniform(r[0], r[1]) for r in self.angle_range]
        else:
            angles = [random.uniform(self.angle_range[0], self.angle_range[1]) for _ in range(a.getImageDimension())]
        # Support per-axis or single range for scale
        if isinstance(self.scale_range[0], (tuple, list)):
            scales = [random.uniform(r[0], r[1]) for r in self.scale_range]
        else:
            scales = [random.uniform(self.scale_range[0], self.scale_range[1]) for _ in range(a.getImageDimension())]
        # Support per-axis or single range for translation
        if isinstance(self.translation_range[0], (tuple, list)):
            translation = [random.uniform(r[0], r[1]) for r in self.translation_range]
        else:
                translation = [random.uniform(self.translation_range[0], self.translation_range[1]) for _ in range(a.getImageDimension())]
        for i, a in enumerate(ables):
            # Determine center: from meta if available, else from image

            # Pad image so all corners remain after transform
            safe_a = SafeImaginable(image=a.getImage())
            safe_a.safepaddingfortransform(angles, center=center, translation=translation)
            # Resample original able to padded target, preserving class
            a.resampleOnTargetImage(safe_a)
            # Create affine matrix
            affine_matrix = create_affine_matrix(rotation=angles, scaling=scales)
            # Apply affine transformation
            a.transformImageAffine(A=affine_matrix.flatten(), translation=translation, center=center)
        return ables
import numpy as np

import pyable.imaginable as ima
class IntensityZScore:
    """
    Z-score normalization for images only (not labelmaps or rois).
    """
    def __call__(self, ables, meta):
        for a in ables:
                if ima.SITKImaginable is not None and isinstance(a, ima.SITKImaginable):
                    # Use Imaginable API for mean and std
                    mean = a.getMeanValue()
                    std = a.getStdValue()
                    if std == 0:
                        std = 1.0
                    # Subtract mean and divide by std using Imaginable methods
                    a = a.getDuplicate()
                    a.subtract(mean)
                    a.divide(std)
                    a.setImageFromNumpy(a.getImageAsNumpy())
        return ables

class IntensityPercentile:
    """
    Percentile normalization for images only (not labelmaps or rois).
    Scales intensities to [0, 1] using given percentiles (default 0, 100).
    """
    def __init__(self, low=0, high=100):
        self.low = low
        self.high = high
    def __call__(self, ables, meta):
        for a in ables:
                if ima.SITKImaginable is not None and isinstance(a, ima.SITKImaginable):
                    arr = a.getImageAsNumpy()
                    mask = arr > 0
                    if np.any(mask):
                        p_low, p_high = np.percentile(arr[mask], [self.low, self.high])
                        
                        if p_high - p_low == 0:
                            # Set all values to zero using Imaginable API
                            a.multiply(0.0)
                        else:
                            a.subtract(p_low)
                            a.divide(p_high - p_low)
                        
        return ables

class IntensityMinMax:
    """
    Rescale image intensities to [0, 1] using the minimum and maximum values of the image (ignoring zeros).
    Only applied to images (not labelmaps or rois).
    """
    def __call__(self, ables, meta):
        for a in ables:
            if ima.SITKImaginable is not None and isinstance(a, ima.SITKImaginable):
                arr = a.getImageAsNumpy()
                mask = arr > 0
                if np.any(mask):
                    vmin = arr[mask].min()
                    vmax = arr[mask].max()
                    if vmax - vmin == 0:
                        a.multiply(0.0)
                    else:
                        a.subtract(vmin)
                        a.divide(vmax - vmin)
        return ables

class SafeImaginable(ima.SITKImaginable):
    def safepaddingfortransform(self, angles, center=None, translation=None, scaling=None):
        """
        Pads the image so that after applying the given affine transform (rotation, scaling, translation),
        all original image corners remain inside the new image bounds.
        Args:
            angles: list of rotation angles in degrees (per axis)
            center: center of rotation/scaling (physical coordinates)
            translation: translation vector (physical coordinates)
            scaling: list of scaling factors (per axis), default [1,1,1]
        """
        import numpy as np
        import SimpleITK as sitk
        from pyable.imaginable import create_affine_matrix

        # Get original corners in physical space
        corners = self.getCornersCoordinates()
        dim = self.getImageDimension()
        if center is None:
            center = self.getImageCenterCoordinate()
        if scaling is None:
            scaling = [1.0] * dim

        # Build affine transform
        affine_matrix = create_affine_matrix(rotation=angles, scaling=scaling)
        if dim == 3:
            transform = sitk.AffineTransform(3)
            transform.SetMatrix(affine_matrix.flatten())
            transform.SetCenter(center)
            if translation is not None:
                transform.SetTranslation(translation)
        elif dim == 2:
            transform = sitk.AffineTransform(2)
            matrix_2d = affine_matrix[:2, :2].flatten()
            transform.SetMatrix(matrix_2d)
            transform.SetCenter(center)
            if translation is not None:
                transform.SetTranslation(translation[:2])
        else:
            raise ValueError("Only 2D/3D supported")

        # Transform all corners
        transformed_corners = [transform.TransformPoint(c) for c in corners]
        all_points = np.array(transformed_corners)

        # Find min/max in each dimension for transformed corners
        min_coords = all_points.min(axis=0)
        max_coords = all_points.max(axis=0)

        # Get current image bounds in physical space
        orig_spacing = np.array(self.getImageSpacing())
        orig_origin = np.array(self.getImageOrigin())
        orig_size = np.array(self.getImageSize())
        img_min = orig_origin
        img_max = orig_origin + orig_spacing * (orig_size - 1)

        # Compute required padding in physical space
        pad_lower_phys = min_coords - img_min
        pad_upper_phys = max_coords - img_max

        # Convert to voxels (round up)
        pad_lower = np.maximum(np.floor(pad_lower_phys / orig_spacing), 0).astype(int)
        pad_upper = np.maximum(np.ceil(pad_upper_phys / orig_spacing), 0).astype(int)

        # Only pad if needed
        if np.any(pad_lower > 0) or np.any(pad_upper > 0):
            self.padImage(pad_lower.tolist(), pad_upper.tolist())
        return self
    


class LabelMapToRoi:
    """
    Transform that extracts specified label values from LabelMapable objects and creates Roiable objects for each value.
    Args:
        labelmapvalues (list): List of label values to extract as ROIs.
    Output:
        Adds the new Roiable objects to the ables list.
    """
    def __init__(self, labelmapvalues,keep_original_labelmap=False):
        self.labelmapvalues = labelmapvalues
        self.keep_original_labelmap = keep_original_labelmap

    def __call__(self, ables, meta):
        new_rois = []
        for a in ables:
            if ima.LabelMapable is not None and isinstance(a, ima.LabelMapable):
                for v in self.labelmapvalues:
                    roi = ima.Roiable(image=(a.getImage() == v))
                    new_rois.append(roi)
        if not self.keep_original_labelmap:
            ables = [a for a in ables if not (ima.LabelMapable is not None and isinstance(a, ima.LabelMapable))]    
        ables.extend(new_rois)
        return ables
    
TRANSFORM_AFTER_RESAMPLING=[
    IntensityPercentile,
    IntensityZScore,
    IntensityMinMax,
]