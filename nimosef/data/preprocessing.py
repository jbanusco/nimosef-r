import numpy as np
import os
import nibabel as nib
import nitransforms as nt
import tqdm
from itertools import product


def compute_bbox(patients, img_root, coords_root, use_roi, session_id='ses-01'):
    """Compute maximum bounding box across all patients."""
    bounding_boxes = []
    for i, subj_id in enumerate(tqdm.tqdm(patients, desc="Computing bounding box"), start=1):
        # if use_roi:
        img_path = os.path.join(img_root, subj_id, session_id, f"{subj_id}_{session_id}_sa.nii.gz")
        # else:
        # img_path = os.path.join(img_root, subj_id, session_id, "anat", f"{subj_id}_{session_id}_acq-HeartSa_Cine.nii.gz")
        
        # Load locl axis and affine transforms from sa_coordinates
        local_axis_path = os.path.join(coords_root, subj_id, session_id, "sa_axis_transform.tfm")
        affine_path = os.path.join(coords_root, subj_id, session_id, "sa_axis_affine.tfm")

        if not (os.path.isfile(img_path) and os.path.isfile(local_axis_path) and os.path.isfile(affine_path)):
            continue

        nii_img = nib.load(img_path)
        local_axis = nt.linear.Affine.from_filename(local_axis_path)._matrix
        affine = nt.linear.Affine.from_filename(affine_path)._matrix

        corners = get_volume_corners(nii_img, affine=affine, local_axis=local_axis)
        subject_bbox = np.max(corners, axis=0) - np.min(corners, axis=0)
        bounding_boxes.append(subject_bbox)

    if not bounding_boxes:
        raise RuntimeError("Could not compute bounding box — no valid patients found.")

    return np.max(bounding_boxes, axis=0)


def normalize_intensity(image_data, low=2, high=98):
    """
    Normalize image intensities to [0, 1] using percentile clipping.

    Args:
        image_data (np.ndarray): raw MRI volume (can be 3D or 4D).
        low (float): lower percentile for clipping (default 2).
        high (float): upper percentile for clipping (default 98).

    Returns:
        np.ndarray: normalized image in [0, 1].
    """
    min_value = np.percentile(image_data, low)
    max_value = np.percentile(image_data, high)
    image_data = np.clip(image_data, min_value, max_value)
    return (image_data - min_value) / (max_value - min_value)


def get_rwc(affine, shape, dx=1, dy=1, dz=1, local_axis=None):
    """
    Compute real-world coordinates (RWC) for each voxel.

    Args:
        affine (np.ndarray): affine transformation matrix (4x4).
        shape (tuple): (x, y, z) shape of the volume.
        dx, dy, dz (int): voxel sampling steps (default: 1).
        local_axis (np.ndarray, optional): 4x4 local axis transform.

    Returns:
        rwc (np.ndarray): [N_vox x 3] real-world coordinates.
        center (np.ndarray): [3] RWC center of the volume.
        coords (np.ndarray): [N_vox x 3] voxel grid indices.
    """
    x, y, z = np.meshgrid(
        np.arange(0, shape[0], dx),
        np.arange(0, shape[1], dy),
        np.arange(0, shape[2], dz),
        indexing="ij"
    )
    coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # voxel indices
    rwc = nib.affines.apply_affine(affine, coords)        # voxel → RWC
    center = nib.affines.apply_affine(affine, np.array(shape) / 2)

    if local_axis is not None:
        rwc = rwc @ local_axis[:3, :3].T + local_axis[:3, 3]
        center = center @ local_axis[:3, :3].T + local_axis[:3, 3]

    return rwc, center, coords


def get_combined_affine(affine, local_axis=None):
    """
    Compose the NIfTI affine with an optional local_axis.

    Parameters
    ----------
    affine : (4,4) np.ndarray
        Global affine (voxel → RWC).
    local_axis : (4,4) np.ndarray or None
        Extra transform (RWC → subject-local).

    Returns
    -------
    combined_affine : (4,4) np.ndarray
        Voxel → local coordinates (if local_axis is given),
        otherwise voxel → RWC.
    inv_affine : (4,4) np.ndarray
        Inverse transform (local → voxel).
    """
    if local_axis is not None:
        # voxel → RWC → local
        combined_affine = local_axis @ affine
    else:
        combined_affine = affine.copy()

    inv_affine = np.linalg.inv(combined_affine)
    return combined_affine, inv_affine


def get_volume_corners(nii_img, affine=None, local_axis=None):
    """
    Compute the 8 physical-space corner coordinates for a NIfTI image.

    Args:
        nii_img (nib.Nifti1Image): NIfTI object.
        affine (np.ndarray, optional): 4x4 affine matrix (default: nii_img.affine).
        local_axis (np.ndarray, optional): 4x4 local axis transform.

    Returns:
        np.ndarray: [8 x 3] array of corner coordinates in RWC.
    """
    shape = nii_img.shape[:3]
    if affine is None:
        affine = nii_img.affine

    # all combinations of min/max indices (0 and shape-1 for each axis)
    indices = list(product(*[[0, s - 1] for s in shape]))
    corners = np.array([nib.affines.apply_affine(affine, idx) for idx in indices])

    if local_axis is not None:
        corners = corners @ local_axis[:3, :3].T + local_axis[:3, 3]

    return corners
