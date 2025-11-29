"""
Tests for pyfe adapter conversion helper
"""
import json
import tempfile
from pathlib import Path

import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "pyable_dataloader.pyfe_adapter",
    Path(__file__).resolve().parents[1] / 'src' / 'pyable_dataloader' / 'pyfe_adapter.py'
)
pyfe_adapter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pyfe_adapter_mod)
convert_manifest_to_pyfe = pyfe_adapter_mod.convert_manifest_to_pyfe


def test_convert_manifest_to_pyfe():
    manifest = {
        'sub001': {
            'images': ['/tmp/img1.nii.gz', '/tmp/img2.nii.gz'],
            'labelmaps': ['/tmp/lm1.nii.gz'],
            'rois': []
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / 'pyfe_manifest.json'
        convert_manifest_to_pyfe(manifest, out)

        with open(out, 'r') as f:
            data = json.load(f)

        assert 'dataset' in data
        assert isinstance(data['dataset'], list)
        assert data['dataset'][0]['id'] == 'sub001'
        assert len(data['dataset'][0]['data']) == 2
        # First entry should include labelmap
        assert 'labelmap' in data['dataset'][0]['data'][0]
