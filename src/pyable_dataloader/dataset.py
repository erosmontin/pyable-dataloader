import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
try:
    from pyable.imaginable import SITKImaginable, LabelMapable,Roiable
except ImportError:
    raise ImportError(
        "pyable not found. Please install it first:\n"
        "  cd pyable && pip install -e ."
    )

import uuid

class PyableDataset(Dataset):
    """
    Simple PyTorch Dataset for medical images using pyable.
    - Each subject must have a reference image (for resampling/cropping)
    - If transforms (augmentation) are set, images/labelmaps are padded before augmentation
    - Resampling to reference is always performed after augmentation (if any)
    - Debug saving as NIfTI is supported
    - Only torch output (numpy in derived class)
    """
    def __init__(
        self,
        manifest,
        transforms=None,
        normalizations=None,
        debug_save_dir=None,
    ):
        if isinstance(manifest, str):
            if manifest.endswith('.json'):
                import json
                with open(manifest, 'r') as f:
                    self.data = json.load(f)
            else:
                raise ValueError(f"Unsupported manifest file type: {manifest}")
        elif isinstance(manifest, dict):
            self.data = manifest
        else:
            raise ValueError(f"Unsupported manifest type: {type(manifest)}")
        self.ids = list(self.data.keys())
        #check which transforms to apply after resampling
        self.transforms = transforms
        self.normalizations = normalizations
        self.debug_save_dir = debug_save_dir
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        subject_id = self.ids[idx]
        item = self.data[subject_id]
        # Load all ables together
        ables = []
        ables += [SITKImaginable(filename=p) for p in item.get('images', [])]
        ables += [LabelMapable(filename=p) for p in item.get('labelmaps', [])]
        ables += [Roiable(filename=p) for p in item.get('rois', [])]

        # Reference image for resampling/cropping
        ref_path = item.get('reference', None)
        if isinstance(ref_path, list):
            ref_path = ref_path[0] if ref_path else None
        reference = SITKImaginable(filename=ref_path) if ref_path else None

        # Compute center (always present in meta)
        cir = item.get('cir_mm', None)
        if cir is None and reference is not None:
            if hasattr(reference, 'getImageCenterCoordinates'):
                cir = np.array(reference.getImageCenterCoordinates())
            else:
                size = np.array(reference.getImageSize())
                idx = (float(size[0]) / 2.0, float(size[1]) / 2.0, float(size[2]) / 2.0)
                cir = reference.getCoordinatesFromIndex(idx)
        meta = {'center': cir, 'subject_id': subject_id}

        # Apply transforms (if any)
        if self.transforms is not None:
            ables = self.transforms(ables, meta)

        # Resample all ables to reference if available
        if reference is not None:
            for a in ables:
                if hasattr(a, 'resampleOnTargetImage'):
                    a = a.resampleOnTargetImage(reference)
        if self.normalizations is not None:
            self.normalizations(ables,meta)
        # Separate ables by type for output
        images = [a for a in ables if isinstance(a, SITKImaginable)]
        labelmaps = [a for a in ables if isinstance(a, LabelMapable)]
        rois = [a for a in ables if isinstance(a, Roiable)]

        def _to_tensor_list(objs):
            out = []
            for o in objs:
                arr = None
                if hasattr(o, 'getImageAsNumpy'):
                    arr = o.getImageAsNumpy()
                elif hasattr(o, 'getNumpy'):
                    arr = o.getNumpy()
                else:
                    arr = np.asarray(o)
                t = torch.tensor(arr, dtype=self.dtype if self.dtype is not None else torch.float32)
                out.append(t)
            return out

        images_tensor_list = _to_tensor_list(images)
        if len(images_tensor_list) > 1:
            images_stacked = torch.stack(images_tensor_list, dim=0)
        elif len(images_tensor_list) == 1:
            images_stacked = images_tensor_list[0].unsqueeze(0)
        else:
            images_stacked = torch.empty((0,))

        labelmaps_tensor = _to_tensor_list(labelmaps)
        rois_tensor = _to_tensor_list(rois)

        # Debug save images, labelmaps, rois if debug_save_dir is set
        if self.debug_save_dir is not None:
            subject_uuid = str(uuid.uuid4())
            # Save images
            for idx, a in enumerate(images):
                fn = f"{self.debug_save_dir}/{subject_uuid}_image_{idx}.nii.gz"
                try:
                    a.writeImageAs(fn)
                except Exception as e:
                    print(f"Debug save failed for image {idx}: {e}")
            # Save labelmaps
            for idx, a in enumerate(labelmaps):
                fn = f"{self.debug_save_dir}/{subject_uuid}_labelmap_{idx}.nii.gz"
                try:
                    a.writeImageAs(fn)
                except Exception as e:
                    print(f"Debug save failed for labelmap {idx}: {e}")
            # Save rois
            for idx, a in enumerate(rois):
                fn = f"{self.debug_save_dir}/{subject_uuid}_roi_{idx}.nii.gz"
                try:
                    a.writeImageAs(fn)
                except Exception as e:
                    print(f"Debug save failed for roi {idx}: {e}")
        sample = {
            'id': subject_id,
            'images': images_stacked,
            'labelmaps': labelmaps_tensor,
            'rois': rois_tensor,
            'meta': meta,
            'uuid': subject_uuid
        }
        return sample
