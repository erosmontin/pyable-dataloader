"""
Tests for new pyable-backed transforms to ensure label preservation
"""
import numpy as np
import pytest
import SimpleITK as sitk

import importlib.util
from pathlib import Path

# Import transforms module directly from src to avoid importing package-level dependencies
spec = importlib.util.spec_from_file_location(
    "pyable_dataloader.transforms",
    Path(__file__).resolve().parents[1] / 'src/pyable_dataloader/transforms.py'
)
transforms_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transforms_mod)

RandomTranslation = transforms_mod.RandomTranslation
RandomRotation = transforms_mod.RandomRotation
RandomAffine = transforms_mod.RandomAffine
RandomBSpline = transforms_mod.RandomBSpline


def create_simple_meta(size=(32, 32, 32), spacing=(1.0, 1.0, 1.0)):
    return {
        'size': list(size),
        'spacing': list(spacing),
        'origin': [0.0, 0.0, 0.0],
        'direction': (1, 0, 0, 0, 1, 0, 0, 0, 1)
    }


def create_labelmap(size=(32, 32, 32)):
    arr = np.zeros(size, dtype=np.uint8)
    center = np.array(size) // 2
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < 6:
                    arr[i, j, k] = 1
    return arr


def test_random_translation_label_preserved():
    labelmap = create_labelmap()
    images = np.zeros((32, 32, 32), dtype=np.float32)
    meta = create_simple_meta()

    transform = RandomTranslation(translation_range=[[-3, 3], [-3, 3], [-3, 3]], prob=1.0)
    _, rois, labelmaps = transform(images, [labelmap], [labelmap], meta)

    # Verify labelmap only contains 0 or 1 (no interpolation values)
    vals = np.unique(labelmaps[0])
    assert all(v in [0, 1] for v in vals), f"Found non-integer labels: {vals}"


def test_random_rotation_label_preserved():
    labelmap = create_labelmap()
    images = np.zeros((32, 32, 32), dtype=np.float32)
    meta = create_simple_meta()

    transform = RandomRotation(rotation_range=[[-5, 5], [-5, 5], [-5, 5]], prob=1.0)
    _, rois, labelmaps = transform(images, [labelmap], [labelmap], meta)

    vals = np.unique(labelmaps[0])
    assert all(v in [0, 1] for v in vals)


def test_random_affine_label_preserved():
    labelmap = create_labelmap()
    images = np.zeros((32, 32, 32), dtype=np.float32)
    meta = create_simple_meta()

    transform = RandomAffine(rotation_range=10.0, zoom_range=(0.9, 1.1), shift_range=3.0, prob=1.0)
    _, rois, labelmaps = transform(images, [labelmap], [labelmap], meta)

    vals = np.unique(labelmaps[0])
    assert all(v in [0, 1] for v in vals)


def test_random_bspline_label_preserved():
    labelmap = create_labelmap()
    images = np.zeros((32, 32, 32), dtype=np.float32)
    meta = create_simple_meta()

    transform = RandomBSpline(mesh_size=(4, 4, 4), magnitude=2.0, prob=1.0)
    _, rois, labelmaps = transform(images, [labelmap], [labelmap], meta)

    vals = np.unique(labelmaps[0])
    assert all(v in [0, 1] for v in vals)
