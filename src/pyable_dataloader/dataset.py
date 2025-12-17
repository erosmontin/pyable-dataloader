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
        debug_save_dir=None,
        debug_save_format='nifti',
        dtype=torch.float32,
        pad_width=10,
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
        self.transforms = transforms
        self.debug_save_dir = debug_save_dir
        self.debug_save_format = debug_save_format
        self.dtype = dtype
        self.pad_width = pad_width

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        subject_id = self.ids[idx]
        item = self.data[subject_id]
        # Support multiple manifest key variants for backward compatibility
        roivalue = item.get('roivalues', item.get('roi_values', item.get('roivalue', 1)))
        labelmapsvalue = item.get('labelmapvalues', item.get('labelmap_values', item.get('labelmapvalue', None)))
        images = [SITKImaginable(filename=p) for p in item.get('images', [])]
        labelmaps = [LabelMapable(filename=p) for p in item.get('labelmaps', [])]
        # Apply filterValues per-labelmap. Support several manifest formats:
        # - flat list of ints: apply same list to every labelmap
        # - list-of-lists: apply entry i to labelmaps[i]
        # - scalar/int: apply that scalar
        if labelmapsvalue is not None:
            normalized_labelmap_values = labelmapsvalue
            # If it's a numpy array, convert to python types for introspection
            try:
                import numpy as _np
                if isinstance(normalized_labelmap_values, _np.ndarray):
                    normalized_labelmap_values = normalized_labelmap_values.tolist()
            except Exception:
                pass

            new_labelmaps = []
            for i, lm in enumerate(labelmaps):
                lv = None
                # If the manifest provided a list-of-lists (per-labelmap), use the i-th entry
                if isinstance(normalized_labelmap_values, (list, tuple)) and len(normalized_labelmap_values) > 0:
                    # Detect flat list of ints (apply same to all) vs list-of-lists
                    first = normalized_labelmap_values[0]
                    if isinstance(first, (list, tuple)) or (hasattr(first, '__iter__') and not isinstance(first, (str, bytes)) and not isinstance(first, int)):
                        # list-of-lists: try to get ith entry
                        if len(normalized_labelmap_values) > i:
                            lv = normalized_labelmap_values[i]
                        else:
                            lv = None
                    else:
                        # flat list of ints: apply to every labelmap
                        lv = normalized_labelmap_values
                else:
                    # scalar or other: pass through
                    lv = normalized_labelmap_values

                try:
                    if lv is not None:
                        new_labelmaps.append(lm.filterValues(lv))
                    else:
                        new_labelmaps.append(lm)
                except Exception:
                    # If filterValues fails for any reason, keep original labelmap
                    new_labelmaps.append(lm)

            labelmaps = new_labelmaps
        rois = [Roiable(filename=p, roivalue=roivalue) for p in item.get('rois', [])]
        
        # Prepare for batch transform pipeline
        
        ref_path = item.get('reference', None)
        if isinstance(ref_path, list):
            ref_path = ref_path[0] if ref_path else None


        if ref_path is not None:
            reference = SITKImaginable(filename=ref_path)
            orig_size = reference.getImageSize()
            # Pad reference image
            pad = self.pad_width if self.transforms is not None else 0
            padded_reference = reference.forkDuplicate()
            if pad > 0:
                lower = [pad] * 3
                upper = [pad] * 3
                padded_reference.padImage(lower, upper)

            # Save padded reference for debug
            if self.debug_save_dir is not None and pad > 0:
                from pathlib import Path
                debug_dir = Path(self.debug_save_dir)
                debug_dir.mkdir(parents=True, exist_ok=True)
                padded_reference.writeImageAs(str(debug_dir / f"{subject_id}_padded_reference.nii.gz"))

            cir = item.get('cir_mm', None)
            # Center for transforms: cir_mm if present, else center of padded reference
            if cir is None:
                if hasattr(padded_reference, 'getImageCenterCoordinates'):
                    cir = np.array(padded_reference.getImageCenterCoordinates())
                else:
                    size = np.array(padded_reference.getImageSize())
                    # Use float indices for TransformContinuousIndexToPhysicalPoint
                    idx = (float(size[0]) / 2.0, float(size[1]) / 2.0, float(size[2]) / 2.0)
                    cir = padded_reference.getCoordinatesFromIndex(idx)
            # Build meta and include any manifest-provided label/roi value hints so
            # transforms (e.g., LabelMapOneHot) can read consistent label lists.
            meta = {'center': cir}
            # Normalize and attach manifest label/roi hints to meta under the
            # canonical keys expected by transforms.
            if labelmapsvalue is not None:
                # If user provided a flat list of ints for a single labelmap, wrap
                # as a list-of-lists so downstream code can index per-labelmap.
                if isinstance(labelmapsvalue, (list, tuple)) and labelmapsvalue and all(isinstance(x, (int, np.integer)) for x in labelmapsvalue):
                    meta['labelmap_values'] = [list(labelmapsvalue)
                                               ]
                else:
                    meta['labelmap_values'] = labelmapsvalue
            else:
                meta['labelmap_values'] = None

            if roivalue is not None:
                meta['roi_values'] = roivalue
            else:
                meta['roi_values'] = None

            # Batch transform pipeline
            if self.transforms is not None:
                transforms = self.transforms
                if not isinstance(transforms, (list, tuple)):
                    transforms = [transforms]
                for t in transforms:
                    images, rois, labelmaps = t(images, rois, labelmaps, meta)
                # Ensure all outputs are still Imaginable/LabelMapable/Roiable
                # Ensure outputs are Imaginable/LabelMapable/Roiable. If a transform
                # returned numpy arrays, convert them back into the appropriate
                # Imaginable wrappers using the padded reference (if available)
                # so the dataset can resample them to the original reference.
                def _ensure_imaginable_list(objs, target_cls, ref_img):
                    out = []
                    for o in objs:
                        # Already an Imaginable-like object?
                        if hasattr(o, 'resampleOnTargetImage'):
                            out.append(o)
                            continue

                        # If it's an object with getImageAsNumpy, extract its array
                        if hasattr(o, 'getImageAsNumpy'):
                            try:
                                arr = o.getImageAsNumpy()
                            except Exception:
                                arr = np.asarray(o)
                        else:
                            arr = np.asarray(o)

                        # Create appropriate Imaginable wrapper and write array
                        try:
                            if target_cls is SITKImaginable:
                                obj = SITKImaginable()
                            elif target_cls is LabelMapable:
                                obj = LabelMapable()
                            else:
                                obj = Roiable()

                            ref = ref_img.getImage() if ref_img is not None else None
                            obj.setImageFromNumpy(arr, refimage=ref)
                            out.append(obj)
                        except Exception:
                            # As a last resort, wrap with SITKImaginable
                            obj = SITKImaginable()
                            try:
                                ref = ref_img.getImage() if ref_img is not None else None
                                obj.setImageFromNumpy(arr, refimage=ref)
                                out.append(obj)
                            except Exception:
                                # If even that fails, raise explicit error
                                raise TypeError(f"Could not convert transformed object to Imaginable: {type(o)}")
                    return out

                # Use padded_reference (if created) as the ref for converting numpy arrays
                ref_for_conversion = padded_reference if 'padded_reference' in locals() else (reference if 'reference' in locals() else None)

                images = _ensure_imaginable_list(images, SITKImaginable, ref_for_conversion)
                labelmaps = _ensure_imaginable_list(labelmaps, LabelMapable, lm)
                rois = _ensure_imaginable_list(rois, Roiable, ref_for_conversion)
            # After all transforms, resample each to the original reference
            images = [img.resampleOnTargetImage(reference) for img in images]
            labelmaps = [lm.resampleOnTargetImage(reference) for lm in labelmaps]
            rois = [roi.resampleOnTargetImage(reference) for roi in rois]

            # Debug save transformed images, labelmaps, rois
            if self.debug_save_dir is not None:
                _s=str(uuid.uuid4()).split('-')[0]
                from pathlib import Path
                debug_dir = Path(self.debug_save_dir)
                debug_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(images):
                    img.writeImageAs(str(debug_dir / f"{_s}_{subject_id}_transformed_image_{i}.nii.gz"))
                for i, lm in enumerate(labelmaps):
                    lm.writeImageAs(str(debug_dir / f"{_s}_{subject_id}_transformed_labelmap_{i}.nii.gz"))
                for i, roi in enumerate(rois):
                    roi.writeImageAs(str(debug_dir / f"{_s}_{subject_id}_transformed_roi_{i}.nii.gz"))
            meta = {
                'subject_id': subject_id,
                'spacing': reference.getImageSpacing(),
                'origin': reference.getImageOrigin(),
                'direction': reference.getImageDirection(),
                'size': reference.getImageSize(),
                'source_paths': {
                    'images': item.get('images', []),
                    'labelmaps': item.get('labelmaps', []),
                    'reference': ref_path
                }
            }
        # Build the returned sample (convert Imaginable objects to numpy arrays)
        def _to_numpy_list(objs):
            out = []
            for o in objs:
                if hasattr(o, 'getImageAsNumpy'):
                    arr = o.getImageAsNumpy()
                    # ensure numeric dtype
                    out.append(np.asarray(arr))
                else:
                    out.append(np.asarray(o))
            return out

        images_np_list = _to_numpy_list(images)
        if len(images_np_list) > 1:
            images_stacked = np.stack(images_np_list, axis=0)
        else:
            images_stacked = images_np_list[0][np.newaxis, ...]

        labelmaps_np = _to_numpy_list(labelmaps)
        rois_np = _to_numpy_list(rois)

        sample = {
            'id': subject_id,
            'images': images_stacked,
            'rois': rois_np,
            'labelmaps': labelmaps_np,
            'meta': meta
        }

        return sample
