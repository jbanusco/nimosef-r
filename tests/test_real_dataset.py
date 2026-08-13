import os
import pytest
from nimosef.data.dataset import NiftiDataset
from nimosef.data.io import load_dataset_manifest

MANIFEST_FILE = "/path/to/test_dataset/derivatives/manifests_nimosef/dataset_manifest.json"

@pytest.mark.realdata
@pytest.mark.skipif(not os.path.exists(MANIFEST_FILE), reason="Real dataset manifest not found")
def test_real_dataset_train_val_test():
    manifest = load_dataset_manifest(MANIFEST_FILE)
    assert "train" in manifest and "val" in manifest and "test" in manifest
    assert isinstance(manifest["bbox"], list)
    print("Loaded dataset manifest with bbox:", manifest["bbox"])

    for mode in ["train", "val", "test"]:
        ds = NiftiDataset(MANIFEST_FILE, mode=mode, use_local_axis=True)
        print(f"Number of {mode} subjects:", len(ds))
        if len(ds) == 0:
            print(f"Skipping {mode}: no subjects in the split.")
            continue  # skip empty splits gracefully
    
        sample = ds[0]
        rwc, t0_data, tgt_data, t, transform_info, idx, slice_indices = sample
        print(f"[{mode}] rwc shape:", rwc.shape)
        print(f"[{mode}] t0 intensity shape:", t0_data[0].shape)
        print(f"[{mode}] target seg unique labels:", tgt_data[1].unique())
        assert rwc.ndim == 2
        assert t0_data[0].shape[0] == rwc.shape[0]
