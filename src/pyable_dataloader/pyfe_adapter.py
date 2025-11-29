"""
Utilities to export pyable_dataloader manifests to pyfe-compatible JSON

This module provides a small helper to convert the internal dataset manifest
format to the simplified minimal format expected by pyfe (theF and related
functions).

The produced JSON has the structure expected by pyfe's `exrtactMyFeatures*`
functions: a dictionary with key 'dataset' mapping to a list of dataset
records. Each record contains an 'id', a 'data' list (per-image entries), and
an optional 'augment' field.

This helper keeps the minimal fields required by pyfe: 'image' and
'labelmap'. For ROIs and more advanced options, the 'groups' list can be
extended by users downstream.
"""
import json
from pathlib import Path
from typing import Union, Dict, Any


def convert_manifest_to_pyfe(input_manifest: Union[str, Dict[str, Any]], output_path: Union[str, Path]):
    """
    Convert a pyable-dataloader manifest to a pyfe minimal manifest.

    Args:
        input_manifest: path to JSON manifest or dictionary loaded in memory
        output_path: path to write the converted JSON file
    """
    if isinstance(input_manifest, str):
        with open(input_manifest, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = input_manifest

    dataset_list = []
    for subj_id, item in manifest.items():
        data_entries = []
        # Pair images with labelmaps if possible
        images = item.get('images', [])
        labelmaps = item.get('labelmaps', [])
        rois = item.get('rois', [])

        # Build per-image entries. If labelmaps not present, fall back to ROI
        for i, img in enumerate(images):
            entry = {'image': img}
            # Prefer explicit labelmap if available, else ROI if available
            lm = None
            if i < len(labelmaps):
                lm = labelmaps[i]
            elif i < len(rois):
                lm = rois[i]
            if lm is not None:
                entry['labelmap'] = lm
                entry['labelmapvalue'] = 1
            else:
                # leave out labelmap; pyfe may require it depending on feature type
                pass
            # Minimal groups list for pyfe
            entry['groups'] = []
            data_entries.append(entry)

        dataset_list.append({'id': subj_id, 'data': data_entries})

    out = {'dataset': dataset_list}
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)

    return output_path
