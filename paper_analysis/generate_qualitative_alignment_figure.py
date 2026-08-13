"""
Generate a qualitative slice-alignment figure for the paper (Sec. 4.2 / Table 1
companion), for Original vs. Rigid-Aligned (ANTs) vs. NIMOSEF-R:

  Row 1: whole-heart (LV+MYO+RV) boundary contours from every short-axis
         slice, overlaid in one 2D view and colored by slice index (base ->
         apex). The connected markers trace each slice's centroid, directly
         visualizing what the CoM Z-continuity metrics in Table 1 quantify:
         a tight, small path means good inter-slice continuity, a scattered
         path means poor continuity.
  Row 2: a reformatted long-axis view (a narrow slab through the stack,
         perpendicular to the short-axis slices): the actual MRI intensity
         image in grayscale (mean-intensity projection over the slab), with
         the segmentation boundary overlaid in the same LV/MYO/RV colors as
         row 1's legend, consistent across all three panels. This makes
         inter-slice misalignment visible directly in image space, as a
         discontinuity ("staircase") in the anatomy itself and in the
         boundary overlay -- e.g. the LV literally splitting into
         disconnected regions under severe misalignment.

sa_slice_alignment's (ANTs) segmentation NIfTI files have scl_slope/scl_inter
just barely off 1.0/0.0, so get_fdata() returns values like 2.999999999998
for what's stored as integer label 3. Always round before casting to int
(load_seg() below) -- a bare `.astype(int)` truncates, silently shifting
every label in the file down by one. Same fix applied in the Table 1 pipeline
(tests/test_slice_alignment.py).

Row 2's NIMOSEF-R panel uses the REAL acquired image and segmentation (same
source as the Original panel) with NIMOSEF-R's learned per-slice rigid
correction applied -- not the network's own reconstruction, to keep "real
anatomy" vs. "network prediction" from confounding the comparison. The
correction is extracted directly from the trained checkpoint and applied via
the same coordinate pipeline generate_results.py uses (get_rwc/
get_combined_affine + the model's correction formula, reimplemented in numpy
here to avoid instantiating a full MultiHeadNetwork). Validated by comparing
against the already-trusted seg_corrected.nii.gz: Dice 0.97-0.99 per label.

Row 2's reformatted long-axis slab is +/-3 voxels (~7mm each side): wide
enough to avoid missing the heart in some slices, narrow enough that local
inter-slice discontinuity (ANTs optimizes local adjacent-slice matching, so
a wide slab averages it away) stays visible.

The slab is taken along Y (collapsing Y, keeping the X-Z plane), centered on
the whole-heart centroid, rather than along X. LV and RV are typically
separated more along X than Y in this dataset's orientation; collapsing X
pools RV in behind/around LV in projection, while collapsing Y keeps LV and
RV at distinct X positions -- a coronal-like reformat with both chambers
visible side by side, each still showing its own staircase.

The representative subject and CoM Z-continuity (2nd difference, 90th
percentile) values below come from tests/test_slice_alignment.py's cached
alignment_comparison CSVs. Update SUBJ/SUBJ_IDX/COM2_VALS together to
re-target a different subject. SUBJ_IDX is that subject's position in
dataset_manifest.json's "train" list (all 105 subjects), which is also its
row index into the checkpoint's shape_code/slice_correction tables.

Output: paper/camera_ready/qualitative_alignment_contour.png
"""
import os
import json
import nibabel as nib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, center_of_mass, map_coordinates

from nimosef.data.preprocessing import get_rwc, get_combined_affine

SUBJ = "085"
SUBJ_IDX = 30  # position in dataset_manifest.json's "train" list
MAX_SLICES = 23  # slice_correction table size = num_subjects * MAX_SLICES (from checkpoint shape)
MAX_TRANSLATION = 0.2  # matches MultiHeadNetwork._get_inverse_corrected_coords default
TIME = 0  # end-diastole
BASE = "/path/to/ACDC_dataset/derivatives"
CKPT_PATH = f"{BASE}/nimosef_v2/muon_riemann_model/experiment_20260701_142855/model.pth"
OUT_DIR = "/path/to/output/paper"
REFORMAT_HALF_WIDTH_VOX = 3  # ~7mm each side at this dataset's in-plane spacing
BBOX = np.array([524.1553327773272, 524.1553327773272, 120.0])

# CoM Z-continuity, 2nd difference, 90th percentile for sub-085 (mm), LV+MYO
# -- from the same pipeline/cache used for Table 1 (tests/generate_table_slice_alignment.py, label=[1,2]).
COM2_VALS = {"Original": 19.21, "Rigid Aligned (ANTs)": 5.31, "NIMOSEF-R": 1.81}

SEG_PATHS = {
    "Original": f"{BASE}/sa_segmentation/sub-{SUBJ}/ses-01/sub-{SUBJ}_sa_seg.nii.gz",
    "Rigid Aligned (ANTs)": f"{BASE}/sa_slice_alignment/sub-{SUBJ}/ses-01/sub-{SUBJ}_sa_seg.nii.gz",
}
IMG_PATHS = {
    "Original": f"{BASE}/sa_roi/sub-{SUBJ}/ses-01/sub-{SUBJ}_ses-01_sa.nii.gz",
    "Rigid Aligned (ANTs)": f"{BASE}/sa_slice_alignment/sub-{SUBJ}/ses-01/sub-{SUBJ}_sa.nii.gz",
}

# LV/MYO/RV legend shared by row 1's contours and row 2's overlay.
LABEL_COLORS = {1: "#d62728", 2: "#2ca02c", 3: "#1f77b4"}  # LV, MYO, RV


def load_seg(path, time=TIME):
    """Always round(), never a bare astype(int) truncation -- see module docstring."""
    return np.round(nib.load(path).get_fdata()).astype(int)[:, :, :, time]


def warp_real_data_with_nimosef_correction(subj_transform_path, seg_path, img_path, shape_xyz):
    """
    Apply NIMOSEF-R's learned per-slice rigid correction (extracted from the
    trained checkpoint) to the REAL acquired segmentation and image for one
    subject, via the same RWC/normalization pipeline used at training time.
    Returns (warped_seg, warped_img), both shape `shape_xyz`.
    """
    transform = json.load(open(subj_transform_path))
    affine = np.array(transform["affine"])
    local_axis = np.array(transform["local_axis"])

    or_seg_data = np.round(nib.load(seg_path).get_fdata()[..., TIME])
    or_img_data = nib.load(img_path).get_fdata()[..., TIME]

    rwc, center, voxel_coords = get_rwc(affine, shape_xyz, dz=1, local_axis=local_axis)
    coords_norm = (rwc - center) / (BBOX / 2)
    z_idx = voxel_coords[:, 2].astype(int)  # exact by construction (arange), safe to truncate

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    table = ckpt["slice_correction.table.weight"].numpy()
    flat_idx = SUBJ_IDX * MAX_SLICES + z_idx
    corr = table[flat_idx]
    theta = corr[:, 0]
    t = np.tanh(corr[:, 1:3]) * MAX_TRANSLATION

    xy_shifted = coords_norm[:, :2] - t
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    mis_x = cos_t * xy_shifted[:, 0] + sin_t * xy_shifted[:, 1]
    mis_y = -sin_t * xy_shifted[:, 0] + cos_t * xy_shifted[:, 1]
    misaligned_norm = np.stack([mis_x, mis_y, coords_norm[:, 2]], axis=-1)

    native_rwc = misaligned_norm * (BBOX / 2) + center
    _, inv_aff = get_combined_affine(affine, local_axis)
    native_voxel_T = nib.affines.apply_affine(inv_aff, native_rwc).T

    warped_seg = map_coordinates(or_seg_data, native_voxel_T, order=0, mode="nearest").reshape(shape_xyz)
    warped_img = map_coordinates(or_img_data, native_voxel_T, order=1, mode="nearest").reshape(shape_xyz)
    return np.round(warped_seg).astype(int), warped_img


def slab_label_projection(vol, c0, half_width, axis):
    """Label-priority projection (LV > RV > MYO) over a narrow slab along `axis`."""
    v = np.moveaxis(vol, axis, 0)
    lo, hi = max(0, c0 - half_width), min(v.shape[0], c0 + half_width + 1)
    slab = v[lo:hi]
    out = np.zeros(v.shape[1:], dtype=int)
    out[(slab == 2).any(axis=0)] = 2
    out[(slab == 3).any(axis=0)] = 3
    out[(slab == 1).any(axis=0)] = 1
    return out


def slab_intensity_projection(vol, c0, half_width, axis):
    """Mean-intensity projection over the same narrow slab along `axis`."""
    v = np.moveaxis(vol, axis, 0)
    lo, hi = max(0, c0 - half_width), min(v.shape[0], c0 + half_width + 1)
    return v[lo:hi].mean(axis=0)


def main():
    vols, img_vols = {}, {}
    spacing_xy = spacing_z = None
    for label, p in SEG_PATHS.items():
        img = nib.load(p)
        vols[label] = load_seg(p)
        spacing_xy = abs(img.affine[0, 0])
        spacing_z = abs(img.affine[2, 2])
    for label, p in IMG_PATHS.items():
        d = nib.load(p).get_fdata()[:, :, :, TIME]
        img_vols[label] = d / (d.max() + 1e-8)

    shape_xyz = vols["Original"].shape
    warped_seg, warped_img = warp_real_data_with_nimosef_correction(
        f"{BASE}/sa_implicit/sub-{SUBJ}/ses-01/sub-{SUBJ}_transform.json",
        SEG_PATHS["Original"], IMG_PATHS["Original"], shape_xyz,
    )
    vols["NIMOSEF-R"] = warped_seg
    img_vols["NIMOSEF-R"] = warped_img / (warped_img.max() + 1e-8)

    labels = ["Original", "Rigid Aligned (ANTs)", "NIMOSEF-R"]
    aspect = spacing_z / spacing_xy
    n_z = vols["Original"].shape[2]
    colormap = plt.get_cmap("viridis", n_z)

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 0.55, 1.05], hspace=0.3)
    row0 = [fig.add_subplot(gs[0, c]) for c in range(3)]
    # Thin colorbar: give the row-1 slot enough height to fit tick labels + xlabel
    # without colliding with row 2, but only use the TOP fraction of that box for
    # the actual bar (a full-height horizontal colorbar there renders as a thick
    # slab, since fig.colorbar() fills its cax's entire box height).
    row1_pos = fig.add_subplot(gs[1, :]).get_position()
    cbar_ax = fig.add_axes([row1_pos.x0, row1_pos.y1 - 0.35 * row1_pos.height,
                             row1_pos.width, 0.35 * row1_pos.height])
    fig.delaxes(fig.axes[-2])  # drop the full-height placeholder subplot, keep only cbar_ax
    row2 = [fig.add_subplot(gs[2, c]) for c in range(3)]

    # --- Row 1: contour overlay + centroid path (whole heart) ---
    for ax, label in zip(row0, labels):
        vol = vols[label]
        centroids = []
        for z in range(vol.shape[2]):
            mask = (vol[:, :, z] != 0)
            if mask.sum() < 5:
                continue
            eroded = binary_erosion(mask)
            boundary = mask & (~eroded)
            ys, xs = np.where(boundary)
            if len(xs) == 0:
                continue
            ax.scatter(xs * spacing_xy, ys * spacing_xy, s=1.2,
                       color=colormap(z / max(n_z - 1, 1)), alpha=0.55, zorder=1)
            cy, cx = center_of_mass(mask)
            centroids.append((z, cx * spacing_xy, cy * spacing_xy))

        centroids = np.array(centroids)
        if len(centroids) > 0:
            ax.plot(centroids[:, 1], centroids[:, 2], '-', color='black', linewidth=1.2, alpha=0.7, zorder=2)
            ax.scatter(centroids[:, 1], centroids[:, 2], c=centroids[:, 0], cmap=colormap,
                       vmin=0, vmax=n_z - 1, s=45, edgecolors='black', linewidths=0.8, zorder=3)

        com2_str = f"{COM2_VALS[label]:.1f} mm" if COM2_VALS[label] is not None else "TBD"
        ax.set_title(f"{label}\nCoM Z-cont. (2nd diff, 90th pct): {com2_str}", fontsize=9.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=n_z - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Slice index (base -> apex)", fontsize=9)

    # --- Row 2: reformatted long-axis view -- grayscale image + colored segmentation contour ---
    for ax, label in zip(row2, labels):
        vol = vols[label]
        c0 = int(round(center_of_mass(vol != 0)[1]))
        label_proj = slab_label_projection(vol, c0, REFORMAT_HALF_WIDTH_VOX, axis=1)
        img_proj = slab_intensity_projection(img_vols[label], c0, REFORMAT_HALF_WIDTH_VOX, axis=1)

        ax.imshow(np.rot90(img_proj), cmap="gray", vmin=0, vmax=img_proj.max(),
                  interpolation="nearest", aspect=aspect, zorder=0)

        for lab_val, color in LABEL_COLORS.items():
            mask = (label_proj == lab_val)
            if mask.sum() < 3:
                continue
            boundary = mask & (~binary_erosion(mask))
            xs, zs = np.where(boundary)  # label_proj has shape (X, Z)
            # np.rot90((X,Z)) -> (Z,X) via B[row,col] = A[col, Z_size-1-row], i.e. row=Z_size-1-z, col=x
            z_size = label_proj.shape[1]
            ax.scatter(xs, z_size - 1 - zs, s=3, color=color, zorder=2)

        ax.set_title("Reformatted long-axis view", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "qualitative_alignment_contour.png")
    plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
