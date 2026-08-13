import os
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
import nitransforms as nt
from sklearn.model_selection import train_test_split
import json

from nimosef.data.preprocessing import normalize_intensity, get_rwc, compute_bbox
from nimosef.data.io import save_parquet_data, save_subject_manifest, save_dataset_manifest, save_subject_arrays_npz
from nimosef.utils.core import str2bool


def preprocess_subject(subj_id, img_path, seg_path, coords_path, bbox, save_root, manifest_dir, session_id='ses-01'):
    """Process one subject, save parquet + subject manifest."""

    manifest_path = os.path.join(manifest_dir, f"{subj_id}_{session_id}.json")
    save_path = os.path.join(save_root, subj_id, session_id)
    
    npz_path = os.path.join(save_path, f"{subj_id}.npz")
    
    parquet_paths = {
        "intensity": os.path.join(save_path, f"{subj_id}_intensity.parquet"),
        "segmentation": os.path.join(save_path, f"{subj_id}_segmentation.parquet"),
        "coords_scaled": os.path.join(save_path, f"{subj_id}_coords_scaled_local.parquet"),
        "coords": os.path.join(save_path, f"{subj_id}_coords_local.parquet"),
        "indices": os.path.join(save_path, f"{subj_id}_indices.parquet"),
        "transform": os.path.join(save_path, f"{subj_id}_transform.json"),
        "npz": npz_path,
    }

    # Always generate manifest for compatibility, but check if we can skip processing
    files_exist = all(os.path.isfile(p) for p in parquet_paths.values())
    if not files_exist:
        # Load NIfTI
        image = nib.load(img_path)
        seg = nib.load(seg_path)
        image_data = normalize_intensity(image.get_fdata())
        seg_data = seg.get_fdata().astype(np.int64)

        # Load transforms
        local_axis_path = os.path.join(coords_path, subj_id, session_id, "sa_axis_transform.tfm")
        affine_path = os.path.join(coords_path, subj_id, session_id, "sa_axis_affine.tfm")
        local_axis = nt.linear.Affine.from_filename(local_axis_path)._matrix
        affine = nt.linear.Affine.from_filename(affine_path)._matrix

        # Real-world coords
        shape = image_data.shape[:3]
        rwc, center, indices = get_rwc(affine, shape, local_axis=local_axis)
        rwc_scaled = (rwc - center) / (np.array(bbox) / 2)  # Divide to put it in ~[-1, 1]

        # Reshape to (N_vox, T)
        num_frames = image_data.shape[-1]
        intensity = pd.DataFrame(
            image_data.reshape(-1, num_frames),
            columns=[f"t{i}" for i in range(num_frames)]
        )
        segmentation = pd.DataFrame(
            seg_data.reshape(-1, num_frames),
            columns=[f"t{i}" for i in range(num_frames)]
        )
        df_coords_scaled = pd.DataFrame(rwc_scaled, columns=["x", "y", "z"])
        df_coords = pd.DataFrame(rwc, columns=["x", "y", "z"])
        df_indices = pd.DataFrame(indices, columns=["x", "y", "z"])

        # Save parquet
        save_parquet_data(
            save_path, subj_id,
            intensity, segmentation,
            df_coords_scaled, df_coords, df_indices,
            center, affine, local_axis
        )

        # Save .npz for fast runtime
        npz_path_saved = save_subject_arrays_npz(
            save_path,
            subj_id,
            intensity,
            segmentation,
            df_coords_scaled,
            df_indices,
        )
        print(f"Saved {npz_path_saved}")
    else:
        print(f"Skipping preprocessing for {subj_id}, files already exist.")

    # Save subject manifest JSON
    nifti_paths = {
        "image": img_path,
        "segmentation": seg_path,
    }
    meta = {"bbox": bbox.tolist()}

    save_subject_manifest(manifest_path, subj_id, parquet_paths, nifti_paths, meta)

    return manifest_path


def main():    
    # data_path="/path/to/test_dataset"
    data_path="/path/to/ACDC_dataset"
    # data_path="/path/to/heartmagic_dataset"

    save_dir="sa_implicit"
    manifest_dir="manifests_nimosef"
    session_id='ses-01'

    parser = argparse.ArgumentParser(description="Preprocess NIfTI subjects into parquet + manifest format")
    parser.add_argument("--root", type=str, required=False, default=data_path,
                        help="Root dataset folder (contains sub-*/ and derivatives/)")
    parser.add_argument("--patients", type=int, default=-1,
                        help="Number of patients to preprocess (default: all)")
    parser.add_argument("--use_roi", type=str2bool, default=True,
                        help="Use ROI images from derivatives/sa_roi instead of full images")
    parser.add_argument("--bbox", type=float, nargs=3, default=None,
                        help="Optional bounding box (x y z). If not set, auto-computed across patients.")
    parser.add_argument("--save_dir", type=str, default=save_dir,
                        help="Directory to save the parquet files and the transforms (inside derivatives/)")
    parser.add_argument("--manifest_dir", type=str, default=manifest_dir,
                        help="Directory to save per-subject JSON manifests (inside derivatives/)")

    args = parser.parse_args()

    derivatives_path = os.path.join(args.root, "derivatives")
    coords_root = os.path.join(derivatives_path, "sa_coordinates")
    if args.use_roi:
        img_root = os.path.join(derivatives_path, "sa_roi")
        seg_root = os.path.join(derivatives_path, "sa_segmentation")
    else:
        img_root = args.root
        seg_root = os.path.join(derivatives_path, "sa_segmentation")

    # Ensure manifest directory exists
    manifest_dir = os.path.join(derivatives_path, args.manifest_dir)
    os.makedirs(manifest_dir, exist_ok=True)

    # Ensure save directory exists
    save_dir = os.path.join(derivatives_path, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    patients = [p for p in os.listdir(img_root) if p.startswith("sub-")]
    if args.patients > 0:
        patients = patients[:args.patients]

    print(f"Found {len(patients)} patients, preprocessing {len(patients)}...")

    # --- bounding box ---
    if args.bbox is None:
        print("Auto-computing bounding box...")
        bbox = compute_bbox(patients, img_root, coords_root, args.use_roi)
    else:
        bbox = np.array(args.bbox)

    print(f"Using bounding box: {bbox}")

    # --- preprocess each subject ---
    manifest_paths = []
    valid_patients = []
    for subj_id in patients:
        # if args.use_roi:            
        img_path = os.path.join(img_root, subj_id, session_id, f"{subj_id}_{session_id}_sa.nii.gz")            
        seg_path = os.path.join(seg_root, subj_id, session_id, f"{subj_id}_sa_seg.nii.gz")
        # else:
        #     img_path = os.path.join(img_root, subj_id, session_id, "anat", f"{subj_id}_{session_id}_acq-HeartSa_Cine.nii.gz")            
        #     seg_path = os.path.join(seg_root, subj_id, session_id, f"{subj_id}_sa_seg_all_corrected.nii.gz")

        if not os.path.isfile(img_path):
            print(f"Skipping {subj_id}: missing image file, {img_path}.")
            continue

        if not os.path.isfile(seg_path):
            print(f"Skipping {subj_id}: missing seg file, {seg_path}.")
            continue

        print(f"Processing {subj_id}...")
        try:
            manifest_path = preprocess_subject(subj_id, img_path, seg_path, coords_root, bbox, save_dir, manifest_dir)
        except Exception as e:
            print(f"Error: {e}")
            continue
        manifest_paths.append(manifest_path)

        # Add to the list of valid patients
        valid_patients.append(subj_id)

    # --- save dataset-level manifest ---
    train, temp = train_test_split(valid_patients, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    split_data = {
        "train": [os.path.join(manifest_dir, f"{sid}_{session_id}.json") for sid in train],
        "val": [os.path.join(manifest_dir, f"{sid}_{session_id}.json") for sid in val],
        "test": [os.path.join(manifest_dir, f"{sid}_{session_id}.json") for sid in test],
    }

    dataset_manifest = os.path.join(manifest_dir, "dataset_manifest.json")
    save_dataset_manifest(dataset_manifest, split_data, bbox.tolist())

    print(f"✅ Preprocessing finished. Dataset manifest saved to {dataset_manifest}")


if __name__ == "__main__":
    main()
