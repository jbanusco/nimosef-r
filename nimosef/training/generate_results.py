import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from scipy.ndimage import map_coordinates

from nimosef.losses.utils import extract_boundary_points
from nimosef.models.nimosef import MultiHeadNetwork
from nimosef.data.dataset import NiftiDataset
from nimosef.training.args import get_inference_parser
from nimosef.data.preprocessing import get_rwc, get_combined_affine
from nimosef.data.io import load_subject_manifest


def __extract_boundary_fast(flat_seg_np, grid_shape, coords_arr):
    """
    Very fast grid-based morphological boundary extraction to replace O(N^2) GPU KNN.
    """
    import scipy.ndimage
    seg_3d = flat_seg_np.reshape(grid_shape)
    boundary_mask = np.zeros_like(seg_3d, dtype=bool)
    
    for c in np.unique(seg_3d):
        if c == 0: continue
        class_mask = (seg_3d == c)
        eroded = scipy.ndimage.binary_erosion(class_mask)
        boundary_mask |= (class_mask & ~eroded)
        
    b_indices = np.where(boundary_mask.ravel())[0]
    b_indices_t = torch.tensor(b_indices, device=coords_arr.device).long()
    boundary_coords_t = coords_arr[b_indices_t]
    
    return b_indices_t, boundary_coords_t


def get_shapes_and_coordinates_synthetic(dataset, res_factor_z, i=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_dim = 128
    y_dim = 128
    z_dim = 10
    time_idx = 50
    pixdim_low_res = (np.asarray(dataset.bbox)/2) / np.asarray([x_dim, y_dim, z_dim])

    lower_lim = -1
    upper_lim = 1
    x, y, z = np.meshgrid(
        np.linspace(lower_lim, upper_lim, x_dim),
        np.linspace(lower_lim, upper_lim, y_dim),
        np.linspace(lower_lim, upper_lim, z_dim),
        indexing="ij"
        )
    coords_original = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # Shape: (N, 3)
    coords_arr_original = torch.tensor(coords_original).float().to(device)

    x, y, z = np.meshgrid(
        np.linspace(lower_lim, upper_lim, x_dim),
        np.linspace(lower_lim, upper_lim, y_dim),
        np.linspace(lower_lim, upper_lim, int(z_dim * res_factor_z)),
        indexing="ij"
        )
    coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # Shape: (N, 3)
    coords_arr = torch.tensor(coords).float().to(device)

    original_shape_xyzt = np.array((x_dim, y_dim, z_dim, time_idx))
    new_shape_xyzt = np.round(original_shape_xyzt * np.array((1, 1, res_factor_z, 1))).astype(int)
    center = np.zeros(3)

    return center, original_shape_xyzt, new_shape_xyzt, coords_arr_original, coords_arr, pixdim_low_res


def get_shape_and_coordinates_subject(dataset, res_factor_z, idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    subj = dataset.subjects[idx]
    patient_id = dataset.patients[idx]

    print(f"Processing subject {idx}: {patient_id}")
    
    # Get img paths    
    subj_manifest = load_subject_manifest(dataset.subject_manifests[idx])
    img_path = subj_manifest["nifti"]["image"]
    #img_path = img_path.replace("/usr/data", "/space_mounts/data/ACDC")
    seg_path = subj_manifest["nifti"]["segmentation"]
    #seg_path = seg_path.replace("/usr/data", "/space_mounts/data/ACDC")

    # Load the iamge
    or_img = nib.load(img_path)
    or_img_data = or_img.get_fdata()
    or_seg_data = nib.load(seg_path).get_fdata()

    affine = or_img.affine
    pixdim_low_res = or_img.header['pixdim'][1:4]  # Should work
    original_shape_xyzt = np.asarray(or_img.shape)[:4]
    new_shape_xyzt = np.round(original_shape_xyzt * np.array((1, 1, res_factor_z, 1))).astype(int)    

    # Get the transform
    loaded_transform = dataset.get_subject_transform(idx)

    # Get real-world coordinates and normalize time
    local_axis = np.array(loaded_transform["local_axis"])
    affine = np.array(loaded_transform["affine"])
    
    # Get the sampling coordinates
    # The dx/dy/dz here are in 'voxels', i.e: how many voxels per step
    rwc, center, _ = get_rwc(affine, original_shape_xyzt[:3], dx=1, dy=1, dz=1/res_factor_z, local_axis=local_axis) # New res.
    rwc_original, center, _ = get_rwc(affine, original_shape_xyzt[:3], dz=1, local_axis=local_axis) # Original res.

    # Center the RWC, subject independent
    scale_factor = np.array(dataset.bbox) / 2
    rwc = (rwc - center) / scale_factor
    rwc_original = (rwc_original - center) / scale_factor

    coords_arr = torch.tensor(rwc).detach().float()
    coords_arr = coords_arr.to(device)

    coords_arr_original = torch.tensor(rwc_original).detach().float()
    coords_arr_original = coords_arr_original.to(device)

    return center, original_shape_xyzt, new_shape_xyzt, coords_arr_original, coords_arr, pixdim_low_res, or_img_data, or_seg_data


def get_indices_from_shape(shape):
    x, y, z = np.meshgrid(
        np.arange(0, shape[0], 1),
        np.arange(0, shape[1], 1),
        np.arange(0, shape[2], 1),
        indexing="ij"
    )
    coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # voxel indices

    return coords


def save_motion_corrected_imgs(model, coords, center, pred_im_final, pred_seg_final,
                               new_shape_xyzt, sample_id, slice_idx,
                               affine, local_axis, bbox):
    """
    Warp reconstructed images/segmentations using slice correction,
    and save as 4D NIfTI volumes (X,Y,Z,T).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_dim, y_dim, z_dim, time_idx = new_shape_xyzt

    # --- Step 1: compute corrected coords (doesn’t depend on time) ---
    corrected_coords = model._get_inverse_corrected_coords(coords, sample_id, slice_idx)
    corrected_coords = corrected_coords.detach().cpu().numpy()

    # Rescale from [-1,1] back to local physical space
    scale_factor = np.array(bbox) / 2
    corrected_coords = corrected_coords * scale_factor
    corrected_coords = corrected_coords + center

    # --- Step 2: undo combined affine (local → voxel indices) ---
    combined_aff, inv_aff = get_combined_affine(affine, local_axis)
    vox_back = nib.affines.apply_affine(inv_aff, corrected_coords)  # (N, 3)

    # --- Step 3: round/clamp for seg, float for intensity ---
    vox_back = vox_back.T  # shape (3, N) for map_coordinates

    # Allocate warped arrays
    warped_im_final = np.zeros_like(pred_im_final, dtype=np.float32)
    warped_seg_final = np.zeros_like(pred_seg_final, dtype=np.int8)

    # --- Step 4: loop over time frames ---
    for t in range(time_idx):
        # Intensity: continuous interpolation
        warped_im_final[..., t] = map_coordinates(
            pred_im_final[..., t],
            vox_back,
            order=1,  # linear interpolation
            mode="nearest"
        ).reshape(x_dim, y_dim, z_dim)

        # Segmentation: nearest neighbor
        seg_vals = map_coordinates(
            pred_seg_final[..., t],
            vox_back,
            order=0,  # nearest
            mode="nearest"
        )
        warped_seg_final[..., t] = seg_vals.reshape(x_dim, y_dim, z_dim)

    return warped_im_final, warped_seg_final


def generate_reconstructed_images(model, val_dataset, save_folder, res_factor_z=1., 
                                  reprocess=False, num_subjects=None, is_synthetic=False,
                                  save_motion_corrected=False):
    """
    Generate the reconstructed images for the validation dataset
    """
    if num_subjects is None:
        num_subjects = len(val_dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Produce some nifti images for the validation dataset:
    print("Generating validation results")

    model.to(device)
    model.eval()

    for i in range(0, num_subjects):
        if i >= num_subjects:
            print(f"Processed {num_subjects} subjects.")
            break

        print(f"Processing subject {i}")
        list_pred_boundaries = []
        list_true_boundaries = []
        list_displacement = []

        patient_id = val_dataset.patients[i]
        save_path = os.path.join(save_folder, f'{patient_id}')
        os.makedirs(save_path, exist_ok=True)
        
        seg_filename = os.path.join(save_path, f"{patient_id}_seg_gt.nii.gz")
        if os.path.isfile(seg_filename) and not reprocess:
            print(f"Already processed!")
            continue
        
        if is_synthetic:
            # Use classic dimensions
            center, original_shape_xyzt, new_shape_xyzt, coords_arr_original, coords_arr, pixdim_low_res = get_shapes_and_coordinates_synthetic(val_dataset, res_factor_z, i=0)
            #skip_correction = False
        else:
            center, original_shape_xyzt, new_shape_xyzt, coords_arr_original, coords_arr, pixdim_low_res, or_img_data, or_seg_data = get_shape_and_coordinates_subject(val_dataset, res_factor_z, i)
            #skip_correction = False

        # Sample ID        
        sample_id = torch.ones(coords_arr.shape[0], device=device)      
        sample_id = (sample_id * i).long()

        sample_id_original = torch.ones(coords_arr_original.shape[0], device=device)
        sample_id_original = (sample_id_original * i).long()

        # Slices ID
        indices_original = get_indices_from_shape(original_shape_xyzt)
        indices_new = get_indices_from_shape(new_shape_xyzt)
        slice_indices_or = torch.tensor(indices_original[:, 2]).long().to(device)
        
        # Map new Z indices back to original Z indices
        # Get original Z-coordinates (physical space)
        orig_z = np.linspace(-1, 1, original_shape_xyzt[2])

        # Get new Z-coordinates (upsampled grid)
        new_z = np.linspace(-1, 1, new_shape_xyzt[2])

        # For each new slice, find the closest original slice index
        z_map = np.array([np.argmin(np.abs(orig_z - z_val)) for z_val in new_z])

        mapped_z = z_map[indices_new[:, 2]]
        slice_indices_new = torch.tensor(mapped_z).long().to(device)

        # Prepare the saving arrays
        pred_im_final = np.zeros(new_shape_xyzt)
        pred_seg_final = np.zeros_like(pred_im_final)
        x_dim, y_dim, z_dim = new_shape_xyzt[:3]
        time_idx = original_shape_xyzt[-1]

        # ==== Extract boundaries from t0 and t predictions using argmax.
        for j in range(0, time_idx):
            # ==== Get prediction at t
            t = j / time_idx                        
            t_tgt = torch.tensor(t).to(device).float()
            t_tgt = torch.tile(t_tgt, (coords_arr.shape[0], 1))
            # pred_seg, pred_im, displacement_t, h_t = model(coords_arr, t_tgt, sample_id, slice_indices_new)
            
            # Predict in chunks to avoid GPU OOM for huge high-res grids
            chunk_size = 50000
            pred_seg_list, pred_im_list, disp_list = [], [], []
            with torch.no_grad():
                for idx in range(0, coords_arr.shape[0], chunk_size):
                    c_chunk = coords_arr[idx : idx + chunk_size]
                    t_chunk = t_tgt[idx : idx + chunk_size]
                    s_chunk = sample_id[idx : idx + chunk_size]
                    idx_chunk = slice_indices_new[idx : idx + chunk_size]
                    
                    seg_c, im_c, disp_c, h_t = model(c_chunk, t_chunk, s_chunk, idx_chunk)
                    pred_seg_list.append(seg_c)
                    pred_im_list.append(im_c)
                    disp_list.append(disp_c)
                    
            pred_seg = torch.cat(pred_seg_list, dim=0)
            pred_im = torch.cat(pred_im_list, dim=0)
            displacement_t = torch.cat(disp_list, dim=0)

            # ==== Boundary tracking [original ground truth]
            if not is_synthetic:
                tgt_segmentation = torch.tensor(or_seg_data[..., j].reshape(-1)).to(device).long()
                
                # Fast grid-based boundary extraction instead of GPU-crashing KNN
                # b_indices_t_gt, boundary_coords_t_gt, _ = extract_boundary_points(tgt_segmentation, coords_arr_original, sample_id_original, k=5)
                b_indices_t_gt, boundary_coords_t_gt = __extract_boundary_fast(tgt_segmentation.cpu().numpy(), original_shape_xyzt[:3], coords_arr_original)
                list_true_boundaries.append(boundary_coords_t_gt.cpu().detach().numpy())

            if j == 0:
                # ==== Get boundary at prediction
                seg_t_labels = torch.argmax(pred_seg, dim=1)
                
                # Fast grid-based boundary extraction
                b_indices_t, boundary_coords_t = __extract_boundary_fast(seg_t_labels.cpu().numpy(), new_shape_xyzt[:3], coords_arr)

                if is_synthetic:
                    b_indices_t0_gt = b_indices_t
                    boundary_coords_t0_gt = boundary_coords_t
                else:
                    b_indices_t0_gt = b_indices_t_gt
                    boundary_coords_t0_gt = boundary_coords_t_gt

            # ==== Predicted 'displaced' t0 boundary
            pred_disp_at_boundary = displacement_t[b_indices_t0_gt]
            predicted_boundary_t = boundary_coords_t0_gt + pred_disp_at_boundary
            list_pred_boundaries.append(predicted_boundary_t.cpu().detach().numpy())
            list_displacement.append(pred_disp_at_boundary.cpu().detach().numpy())
            
            # === Back to image
            # Segmentation
            pred_seg = F.softmax(pred_seg, dim=1)
            pred_seg_t = torch.argmax(pred_seg, axis=1)
            
            # Reshape the target data also
            # Use voxel indices to place predictions            
            pred_im_final[indices_new[:, 0], indices_new[:, 1], indices_new[:, 2], j] = pred_im.detach().cpu().numpy().squeeze()
            pred_seg_final[indices_new[:, 0], indices_new[:, 1], indices_new[:, 2], j] = pred_seg_t.detach().cpu().numpy().squeeze()

        # ===== Boundary
        if not is_synthetic:
            df_true_boundaries = pd.concat(
                [pd.DataFrame(boundary, columns=["x", "y", "z"]).assign(time=t)
                    for t, boundary in enumerate(list_true_boundaries)],
                    ignore_index=True
                )
            df_true_boundaries.to_parquet(os.path.join(save_path, f"{patient_id}_true_boundaries.parquet"))
        
        df_pred_boundaries = pd.concat(
            [pd.DataFrame(boundary, columns=["x", "y", "z"]).assign(time=t)
                for t, boundary in enumerate(list_pred_boundaries)],
                ignore_index=True
            )
        df_pred_boundaries.to_parquet(os.path.join(save_path, f"{patient_id}_pred_boundaries.parquet"))

        df_displacement = pd.concat(
            [pd.DataFrame(displacement, columns=["x", "y", "z"]).assign(time=t)
                for t, displacement in enumerate(list_displacement)],
                ignore_index=True
            )
        df_displacement.to_parquet(os.path.join(save_path, f"{patient_id}_displacement.parquet"))

        # === Back to img                
        ratio_res = (np.asarray(original_shape_xyzt) / np.asarray(new_shape_xyzt))[:3]
        pixdim_high_res = pixdim_low_res * ratio_res
        
        # Create the new affine transformation
        new_affine = np.eye(4)
        new_affine[:3, :3] = np.diag(pixdim_high_res)

        old_affine = np.eye(4)
        old_affine[:3, :3] = np.diag(pixdim_low_res)

        #NOTE: We could use the original affine, but in this case we want to compare just the change in resolution

        nifti_seg = nib.Nifti1Image(pred_seg_final.astype(np.int8), new_affine)
        nib.save(nifti_seg, os.path.join(save_path, f"{patient_id}_seg.nii.gz"))

        nifti_image = nib.Nifti1Image(pred_im_final.astype(np.float32), new_affine)
        nib.save(nifti_image, os.path.join(save_path, f"{patient_id}_rec.nii.gz"))

        if not is_synthetic:
            nifti_img = nib.Nifti1Image(or_img_data.astype(np.float32), old_affine)
            nib.save(nifti_img, os.path.join(save_path, f"{patient_id}_im_gt.nii.gz"))

            nifti_seg_or = nib.Nifti1Image(or_seg_data.astype(np.int8), old_affine)
            nib.save(nifti_seg_or, os.path.join(save_path, f"{patient_id}_seg_gt.nii.gz"))

        print(f"Saved images for subject {patient_id} at {save_path}")

        if save_motion_corrected:
            print("Warping the images after correction.")
            
            loaded_transform = val_dataset.get_subject_transform(i)
            local_axis = np.array(loaded_transform["local_axis"])
            affine = np.array(loaded_transform["affine"])

            warped_im_np, warped_seg_np = save_motion_corrected_imgs(
                model=model,
                center=center,
                coords=coords_arr,                  # original [-1,1] coords
                pred_im_final=pred_im_final,        # (X,Y,Z,T) intensities
                pred_seg_final=pred_seg_final,      # (X,Y,Z,T) seg labels
                new_shape_xyzt=new_shape_xyzt,
                sample_id=sample_id,
                slice_idx=slice_indices_new,
                affine=affine,
                local_axis=local_axis,
                bbox=val_dataset.bbox
            )

            # Save warped outputs
            nib.save(nib.Nifti1Image(warped_im_np.astype(np.float32), new_affine), os.path.join(save_path, f"{patient_id}_rec_corrected.nii.gz"))
            nib.save(nib.Nifti1Image(warped_seg_np.astype(np.int8), new_affine), os.path.join(save_path, f"{patient_id}_seg_corrected.nii.gz"))
            print(f"Saved corrected images for subject {patient_id}")

    print("Validation results saved.")



def main():
    parser = get_inference_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dataset ===
    dataset = NiftiDataset(args.split_file, mode=args.mode, preload_data=False)

    # === Build subject_slices mapping ===
    subject_slices = {}
    for subj_idx, subj in enumerate(dataset.subjects):
        subject_slices[subj_idx] = dataset.get_subject_num_slices(subj_idx)

    # === Model ===
    model = MultiHeadNetwork(
        subject_slices,
        num_subjects=len(dataset),
        num_labels=args.num_labels,
        latent_size=args.latent_size,
        motion_size=args.motion_size,
        hidden_size=args.hidden_size,
        num_res_layers=args.num_res_layers,
        linear_head=args.linear_head,
        skip_correction=args.skip_correction,
        riemannian=getattr(args, "riemannian", True)
    )
    model.to(device)

    # === Load pretrained weights ===
    ckpt = torch.load(args.model_to_rec, map_location=device, weights_only=False)  # Check this -- Related to LossTracker?
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt        
    model.load_state_dict(state_dict, strict=False)

    # === Save folder ===
    if args.save_rec_folders is not None:
        save_folder = args.save_rec_folders
    else:
        save_folder = os.path.join(args.save_folder, "reconstructions")
    os.makedirs(save_folder, exist_ok=True)

    # === Run generation ===
    generate_reconstructed_images(
        model=model,
        val_dataset=dataset,
        save_folder=save_folder,
        res_factor_z=args.res_factor_z,
        reprocess=args.overwrite_imgs,
        is_synthetic=False,
        save_motion_corrected=args.save_motion_corrected,
    )


if __name__ == "__main__":
    main()