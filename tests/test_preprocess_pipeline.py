import os
import numpy as np
import nibabel as nib
import pandas as pd
import tempfile
import json
import torch

from nimosef.data.preprocessing import normalize_intensity, get_rwc, get_volume_corners
from nimosef.data.io import (
    save_parquet_data,
    load_parquet_from_manifest,
    save_subject_manifest,
    load_subject_manifest,
    save_dataset_manifest,
    load_dataset_manifest,
    load_nifti,
    load_segmentation
)
from nimosef.data.dataset import NiftiDataset


def test_normalize_intensity():
    arr = np.array([0, 5, 10, 100, 200], dtype=float)
    normed = normalize_intensity(arr)
    assert np.allclose(normed.min(), 0.0)
    assert np.allclose(normed.max(), 1.0)


def test_get_rwc_and_volume_corners():
    affine = np.eye(4)
    shape = (4, 4, 4)
    rwc, center, coords = get_rwc(affine, shape)
    assert rwc.shape[1] == 3
    assert coords.shape[1] == 3
    nii = nib.Nifti1Image(np.zeros(shape), affine)
    corners = get_volume_corners(nii, affine=affine)
    assert corners.shape == (8, 3)


def test_get_rwc_with_local_axis():
    affine = np.eye(4)
    shape = (2, 2, 2)
    # Local axis: simple translation
    local_axis = np.eye(4)
    local_axis[:3, 3] = [10, 0, 0]
    rwc, center, coords = get_rwc(affine, shape, local_axis=local_axis)
    assert np.allclose(rwc[:, 0].min(), 10.0)  # shifted along x
    assert center[0] == 11.0


def test_load_nifti_and_segmentation(tmp_path):
    data = np.zeros((4,4,4))
    seg = np.ones((4,4,4), dtype=np.int16)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), tmp_path/"img.nii.gz")
    nib.save(nib.Nifti1Image(seg, affine), tmp_path/"seg.nii.gz")

    img, aff1 = load_nifti(tmp_path/"img.nii.gz")
    seg_data, aff2 = load_segmentation(tmp_path/"seg.nii.gz")

    assert img.shape == (4,4,4)
    assert seg_data.dtype == np.int64
    assert np.allclose(aff1, affine)
    assert np.allclose(aff2, affine)


def test_dataset_manifest_roundtrip(tmp_path):
    dataset_manifest = tmp_path / "dataset_manifest.json"
    split_dict = {
        "train": ["sub-0001.json"],
        "val": ["sub-0002.json"],
        "test": ["sub-0003.json"],
    }
    bbox = [64, 64, 64]

    # Save
    save_dataset_manifest(dataset_manifest, split_dict, bbox)

    # Load
    loaded = load_dataset_manifest(dataset_manifest)

    # Checks
    assert set(loaded.keys()) == {"train", "val", "test", "bbox"}
    assert loaded["bbox"] == bbox
    assert "sub-0001.json" in loaded["train"]
    assert "sub-0002.json" in loaded["val"]
    assert "sub-0003.json" in loaded["test"]


def test_parquet_and_manifest(tmp_path):
    # Fake subject data
    subj_id = "sub-0001"
    N, T = 10, 2
    df_int = pd.DataFrame(np.random.rand(N, T), columns=[f"t{i}" for i in range(T)])
    df_seg = pd.DataFrame(np.random.randint(0, 3, size=(N, T)), columns=[f"t{i}" for i in range(T)])
    df_coords_scaled = pd.DataFrame(np.random.rand(N, 3), columns=["x", "y", "z"])
    df_coords = pd.DataFrame(np.random.rand(N, 3), columns=["x", "y", "z"])
    df_indices = pd.DataFrame(np.arange(N*3).reshape(N, 3), columns=["x", "y", "z"])
    center, affine, local_axis = np.zeros(3), np.eye(4), np.eye(4)

    save_dir = tmp_path / subj_id
    save_parquet_data(save_dir, subj_id, df_int, df_seg, df_coords_scaled, df_coords, df_indices, center, affine, local_axis)

    # Create subject manifest
    parquet_paths = {
        "intensity": str(save_dir / f"{subj_id}_intensity.parquet"),
        "segmentation": str(save_dir / f"{subj_id}_segmentation.parquet"),
        "coords_scaled": str(save_dir / f"{subj_id}_coords_scaled_local.parquet"),
        "coords": str(save_dir / f"{subj_id}_coords_local.parquet"),
        "indices": str(save_dir / f"{subj_id}_indices.parquet"),
        "transform": str(save_dir / f"{subj_id}_transform.json"),
    }
    nifti_paths = {"image": "/tmp/fake_img.nii.gz", "segmentation": "/tmp/fake_seg.nii.gz"}
    meta = {"bbox": [32, 32, 32]}
    manifest_path = tmp_path / f"{subj_id}.json"
    save_subject_manifest(manifest_path, subj_id, parquet_paths, nifti_paths, meta)

    # Load parquet + manifest
    manifest = load_subject_manifest(manifest_path)
    df_int2, df_seg2, _, _, transform = load_parquet_from_manifest(manifest)
    assert df_int2.shape == df_int.shape
    assert df_seg2.shape == df_seg.shape
    assert "affine" in transform



def test_dataset_end_to_end(tmp_path):
    # Minimal dataset manifest with one subject
    subj_id = "sub-0001"
    manifest_path = tmp_path / f"{subj_id}.json"
    parquet_paths = {
        "intensity": str(tmp_path / f"{subj_id}_intensity.parquet"),
        "segmentation": str(tmp_path / f"{subj_id}_segmentation.parquet"),
        "coords_scaled": str(tmp_path / f"{subj_id}_coords_scaled_local.parquet"),
        "coords": str(tmp_path / f"{subj_id}_coords_local.parquet"),
        "indices": str(tmp_path / f"{subj_id}_indices.parquet"),
        "transform": str(tmp_path / f"{subj_id}_transform.json"),
    }
    # Fake data saved
    N, T = 5, 2
    pd.DataFrame(np.random.rand(N, T), columns=[f"t{i}" for i in range(T)]).to_parquet(parquet_paths["intensity"])
    pd.DataFrame(np.random.randint(0, 2, size=(N, T)), columns=[f"t{i}" for i in range(T)]).to_parquet(parquet_paths["segmentation"])
    pd.DataFrame(np.random.rand(N, 3), columns=["x", "y", "z"]).to_parquet(parquet_paths["coords_scaled"])
    pd.DataFrame(np.arange(N*3).reshape(N, 3), columns=["x", "y", "z"]).to_parquet(parquet_paths["indices"])
    with open(parquet_paths["transform"], "w") as f:
        json.dump({"center": [0,0,0], "affine": np.eye(4).tolist(), "local_axis": np.eye(4).tolist()}, f)

    save_subject_manifest(manifest_path, subj_id, parquet_paths, {"image":"fake","segmentation":"fake"}, {"bbox":[32,32,32]})
    dataset_manifest = tmp_path / "dataset_manifest.json"
    save_dataset_manifest(dataset_manifest, {"train":[str(manifest_path)],"val":[],"test":[]}, [32,32,32])

    ds = NiftiDataset(dataset_manifest, mode="train")
    sample = ds[0]
    rwc, t0_data, tgt_data, t, transform_info, idx, slice_indices = sample
    assert isinstance(rwc, torch.Tensor)

    # Verify consistent voxel counts    
    N = rwc.shape[0]
    assert t0_data[0].shape[0] == N
    assert tgt_data[0].shape[0] == N
