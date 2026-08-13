from scipy.ndimage import rotate
import os
import nibabel as nib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.ndimage import binary_erosion, center_of_mass
from scipy.spatial import cKDTree


def load_data(subject_id, base_path, use_uncorrected=False):

    subject_id_str = f"sub-{subject_id}"
    
    # Try different possible filename constructions
    if use_uncorrected:
        candidates = [
            os.path.join(base_path, subject_id_str, f"{subject_id_str}_seg.nii.gz"),
            os.path.join(base_path, subject_id_str, 'ses-01', f"{subject_id_str}_seg.nii.gz")
        ]
    else:
        candidates = [
            os.path.join(base_path, subject_id_str, 'ses-01', f"{subject_id_str}_sa_seg.nii.gz"),
            os.path.join(base_path, subject_id_str, 'ses-01', f"{subject_id_str}_seg.nii.gz"),
            os.path.join(base_path, subject_id_str, f"{subject_id_str}_seg_corrected.nii.gz"),
            os.path.join(base_path, subject_id_str, f"{subject_id_str}_seg.nii.gz")
        ]
    
    filename = None
    for candidate in candidates:
        if os.path.exists(candidate):
            filename = candidate
            break
            
    if filename is None:
        # Fallback to the default structure
        filename = candidates[0]

    img = nib.load(filename)
    affine = img.affine
    # round(), not a bare astype(int) truncation: some sources (e.g.
    # sa_slice_alignment) store NIfTI scl_slope/scl_inter just barely off
    # 1.0/0.0, so integer labels come back from get_fdata() as e.g.
    # 2.999999999998, and truncation silently shifts every label down by one
    # across the whole volume.
    seg = np.round(img.get_fdata()).astype(int)
    seg = seg[:,:,1:seg.shape[2]-1,1]  #Choose time frame (maybe for loop to do each)
    num_slices = seg.shape[2]

    return seg, num_slices, affine

#def compute_center_mass(base_path, subject_id, label):
def compute_center_mass_dists(label, subject_id, base_path, use_uncorrected=False):

    seg, num_slices, affine = load_data(subject_id, base_path, use_uncorrected=use_uncorrected)
    cofm_whole_others = []
    center_mass_slices = []

    #------ Compute center of mass whole heart ----
    mask_cofm_whole = np.isin(seg[:,:,:], label)
    cofmass_whole = center_of_mass(mask_cofm_whole)
    cofmass_whole = nib.affines.apply_affine(affine, cofmass_whole)

    for slice in range(num_slices):

        mask_cofm = np.isin(seg[:,:,slice], label)

        cofmass = center_of_mass(mask_cofm)
        cofmass = np.array([cofmass[0], cofmass[1], slice])
        cofmass = nib.affines.apply_affine(affine,cofmass)

        center_mass_slices.append(cofmass)

        #------ Distance between whole heart center of mass and remaining slices -------
        cofm_whole_others.append(np.sqrt((cofmass[0]-cofmass_whole[0])**2+
                                         (cofmass[1]-cofmass_whole[1])**2))

    center_mass_slices = np.array(center_mass_slices)

    #----- Distance between center of mass of middle slice and remaining slicses -----
    if num_slices % 2 == 0:
        mid1 = (num_slices//2) - 1
        mid2 = (num_slices//2)
        center_mass_mid = (center_mass_slices[mid1]+center_mass_slices[mid2])/2
    else:
        middle_slice = (num_slices//2)
        center_mass_mid = center_mass_slices[middle_slice]

    cofm_middle_others = np.sqrt((center_mass_slices[:,0]-center_mass_mid[0])**2+
                                 (center_mass_slices[:,1]-center_mass_mid[1])**2)


    return cofmass_whole, center_mass_slices, cofm_middle_others, cofm_whole_others


def compute_displacements_slices(label, subject_id, base_path, use_uncorrected=False):

    seg, num_slices, affine = load_data(subject_id, base_path, use_uncorrected=use_uncorrected)
    _, center_mass_slices, _, _ = compute_center_mass_dists(label, subject_id, base_path, use_uncorrected=use_uncorrected)

    disp_xy_all = []
    mean_disp = []
    dist_cofmass_outline = []
    adjacent_dice_all = []
    normal_similarities_all = []

    #--------- Compute displacements between consecutive slices ------------

    for slice in range(0,num_slices-1):

        contour_k1 = np.zeros((seg.shape[0], seg.shape[1]))
        contour_k2 = np.zeros((seg.shape[0], seg.shape[1]))

        #Binary mask and erosion -- np.isin(seg, label) rather than seg == label,
        # since `label` may be a multi-element list (e.g. LV+MYO combined).
        mask_k1 = np.isin(seg[:,:,slice], label)
        mask_k2 = np.isin(seg[:,:,(slice+1)], label)
        
        # Compute adjacent slice overlap (Dice index)
        intersection = np.logical_and(mask_k1, mask_k2).sum()
        dice = (2.0 * intersection) / (mask_k1.sum() + mask_k2.sum() + 1e-8)
        adjacent_dice_all.append(dice)
        
        erosion_k1 = binary_erosion(mask_k1)
        erosion_k2 = binary_erosion(mask_k2)

        #Contour (intersection of mask and ~erosion) and retrieve indices
        contour_k1 = mask_k1 & (~erosion_k1)
        contour_k2 = mask_k2 & (~erosion_k2)
        idx_contour_k1 = np.array(np.where(contour_k1 == 1)).T
        idx_contour_k2 = np.array(np.where(contour_k2 == 1)).T

        if idx_contour_k1.shape[0] == 0 or idx_contour_k2.shape[0] == 0:
            print(f"Sub-{subject_id}: slice {slice} or {slice+1} is empty!")
            disp_xy_all.append(np.array([0.0]))
            mean_disp.append(0.0)
            dist_cofmass_outline.append(np.array([0.0]))
            normal_similarities_all.append(np.array([1.0]))
            continue

        #Add slice coordinate and perform affine
        vol_k1 = np.column_stack((idx_contour_k1[:,0], idx_contour_k1[:,1],
                                                                            np.full((len(idx_contour_k1),1), slice)))
        vol_k2 = np.column_stack((idx_contour_k2[:,0], idx_contour_k2[:,1],
                                                                            np.full((len(idx_contour_k2),1), slice+1)))
        rwc_contour_k1 = nib.affines.apply_affine(affine, vol_k1)
        rwc_contour_k2 = nib.affines.apply_affine(affine, vol_k2)

        #Query nn and retrieve list of nn
        nn_tree = cKDTree(rwc_contour_k1)
        _, idxs = nn_tree.query(rwc_contour_k2, k=[1])
        idxs_flat = idxs.flatten()
        nn_list = np.squeeze(rwc_contour_k1[idxs])  #(nn,3)

        #L2 distance wrt x and y
        disp_xy = np.sqrt((nn_list[:,0]-rwc_contour_k2[:,0])**2+
                        (nn_list[:,1]-rwc_contour_k2[:,1])**2)

        disp_xy_all.append(disp_xy)
        mean_disp.append(np.mean(disp_xy))

        #L2 distance between cofmass and outline
        if slice == (num_slices-2):
            disp_cofm = np.sqrt((center_mass_slices[slice][0]-rwc_contour_k2[:,0])**2+
                        (center_mass_slices[slice][1]-rwc_contour_k2[:,1])**2)
            dist_cofmass_outline.append(disp_cofm)
        else:
            disp_cofm = np.sqrt((center_mass_slices[slice][0]-rwc_contour_k1[:,0])**2+
                        (center_mass_slices[slice][1]-rwc_contour_k1[:,1])**2)
            dist_cofmass_outline.append(disp_cofm)

        # Compute surface normal alignment consistency
        grad_y1, grad_x1 = np.gradient(mask_k1.astype(float))
        grad_y2, grad_x2 = np.gradient(mask_k2.astype(float))

        y2 = idx_contour_k2[:, 0]
        x2 = idx_contour_k2[:, 1]
        v2 = np.column_stack((grad_x2[y2, x2], grad_y2[y2, x2]))
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8
        v2 = v2 / v2_norm

        y1 = idx_contour_k1[idxs_flat, 0]
        x1 = idx_contour_k1[idxs_flat, 1]
        v1 = np.column_stack((grad_x1[y1, x1], grad_y1[y1, x1]))
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8
        v1 = v1 / v1_norm

        cos_sim = np.sum(v1 * v2, axis=1)
        normal_similarities_all.append(cos_sim)

    return disp_xy_all, mean_disp, dist_cofmass_outline, adjacent_dice_all, normal_similarities_all


def visualize_dists(label, subjects, base_path, is_png, is_scatter, is_plot, cache_dir=None, use_uncorrected=False):    
    import os

    # 1. Determine cache directory and file path
    if cache_dir is None:
        cache_dir = '/path/to/ACDC_dataset/derivatives/nimosef_v2/alignment_comparison'
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        # Fallback to local directory inside workspace
        fallback_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'derivatives_cache')
        print(f"Warning: Could not create/access cache directory {cache_dir}. Falling back to {fallback_dir}. Error: {e}")
        cache_dir = fallback_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    base_name = os.path.basename(os.path.normpath(base_path))
    label_str = '_'.join(map(str, label)) if isinstance(label, (list, tuple, np.ndarray)) else str(label)
    plot_type = 'scatter' if is_scatter else 'violin'
    if use_uncorrected:
        plot_type += '_uncorrected'
    cache_file = os.path.join(cache_dir, f'{base_name}_label_{label_str}_{plot_type}.csv')

    # 2. Try to load existing cached data
    cached_df = None
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file, dtype={'Subject': str})
            print(f"Loaded cached data from {cache_file}")
        except Exception as e:
            print(f"Warning: Could not read cache file {cache_file}: {e}")

    # Identify subjects that need computation
    subjects_to_compute = []
    if cached_df is not None:
        cached_subjects = set(cached_df['Subject'].unique())
        for subject_id in subjects:
            if str(subject_id) not in cached_subjects:
                subjects_to_compute.append(subject_id)
    else:
        subjects_to_compute = list(subjects)

    # 3. Compute for subjects not in cache
    if subjects_to_compute:
        new_records = []
        print(f"Computing alignment metrics for {len(subjects_to_compute)} subjects...")
        for subject_id in subjects_to_compute:
            try:
                _, center_mass_slices, cofm_middle_others, cofm_whole_others = compute_center_mass_dists(label, subject_id, base_path, use_uncorrected=use_uncorrected)
                disp_xy_all, _, dist_cofmass_outline, adjacent_dice_all, normal_similarities_all = compute_displacements_slices(label, subject_id, base_path, use_uncorrected=use_uncorrected)

                if is_scatter:
                    for value in cofm_whole_others:
                        new_records.append({
                            "Subject": str(subject_id),
                            "Metric": "Dist_cofm_whole_others",
                            "Value": value
                        })
                    for value in cofm_middle_others:
                        new_records.append({
                            "Subject": str(subject_id),
                            "Metric": "Dist_cofm_middle_others",
                            "Value": value
                        })
                    # Compute CoM Z-continuity first and second differences
                    if len(center_mass_slices) > 1:
                        diff_x1 = np.diff(center_mass_slices[:, 0])
                        diff_y1 = np.diff(center_mass_slices[:, 1])
                        com_diff1 = np.sqrt(diff_x1**2 + diff_y1**2)
                        for value in com_diff1:
                            new_records.append({
                                "Subject": str(subject_id),
                                "Metric": "CoM_diff1",
                                "Value": value
                            })
                    if len(center_mass_slices) > 2:
                        diff_x2 = np.diff(center_mass_slices[:, 0], n=2)
                        diff_y2 = np.diff(center_mass_slices[:, 1], n=2)
                        com_diff2 = np.sqrt(diff_x2**2 + diff_y2**2)
                        for value in com_diff2:
                            new_records.append({
                                "Subject": str(subject_id),
                                "Metric": "CoM_diff2",
                                "Value": value
                            })
                else:
                    for disp in disp_xy_all:
                        for value in disp:
                            new_records.append({
                                "Subject": str(subject_id),
                                "Metric": "Disp_xy",
                                "Value": value
                            })
                    for dist in dist_cofmass_outline:
                        for value in dist:
                            new_records.append({
                                "Subject": str(subject_id),
                                "Metric": "Dist_cofmass",
                                "Value": value
                            })
                    for value in adjacent_dice_all:
                        new_records.append({
                            "Subject": str(subject_id),
                            "Metric": "Contour_Dice",
                            "Value": value
                        })
                    for sims in normal_similarities_all:
                        for value in sims:
                            new_records.append({
                                "Subject": str(subject_id),
                                "Metric": "Normal_Similarity",
                                "Value": value
                            })
            except Exception as e:
                print(f"Error computing displacements/dists for subject {subject_id}: {e}")

        # If new data was computed, update cache
        if new_records:
            new_df = pd.DataFrame(new_records)
            if cached_df is not None:
                cached_df = pd.concat([cached_df, new_df], ignore_index=True)
            else:
                cached_df = new_df
            
            try:
                cached_df.to_csv(cache_file, index=False)
                print(f"Saved updated cache to {cache_file}")
            except Exception as e:
                print(f"Warning: Could not save cache file {cache_file}: {e}")

    # 4. Filter and prepare the output dataframes
    if cached_df is not None:
        subjects_str = [str(s) for s in subjects]
        res_df = cached_df[cached_df['Subject'].isin(subjects_str)].reset_index(drop=True)
    else:
        res_df = pd.DataFrame(columns=["Subject", "Metric", "Value"])

    if is_scatter:
        df_scatter = res_df
        df_violin = pd.DataFrame(columns=["Subject", "Metric", "Value"])
    else:
        df_scatter = pd.DataFrame(columns=["Subject", "Metric", "Value"])
        df_violin = res_df
    #print(df.head())
    #print(df.size)

    if is_scatter and is_plot:
        # -----Scatter Plots-----
        plt.figure("Center of Mass Distances and Z-Continuity Smoothness", figsize=(12, 10))
        plt.subplot(221)
        plt.plot(df_scatter[df_scatter["Metric"]=="Dist_cofm_whole_others"]["Subject"],
                df_scatter[df_scatter["Metric"]=="Dist_cofm_whole_others"]["Value"],
                'o', markersize=3)
        plt.title("Whole Heart <-> Individual Slices")
        plt.xlabel("Subjects")
        plt.ylabel("Distances (mm)")
        plt.tick_params(axis='x',labelsize='small', labelrotation=90)

        plt.subplot(222)
        plt.plot(df_scatter[df_scatter["Metric"]=="Dist_cofm_middle_others"]["Subject"],
                df_scatter[df_scatter["Metric"]=="Dist_cofm_middle_others"]["Value"],
                'o', markersize=3)
        plt.title("Middle Slice <-> Individual Slices")
        plt.xlabel("Subjects")
        plt.ylabel("Distances (mm)")
        plt.tick_params(axis='x', labelsize='small', labelrotation=90)

        plt.subplot(223)
        plt.plot(df_scatter[df_scatter["Metric"]=="CoM_diff1"]["Subject"],
                df_scatter[df_scatter["Metric"]=="CoM_diff1"]["Value"],
                'o', markersize=3)
        plt.title("CoM Z-Continuity: First-order Diff")
        plt.xlabel("Subjects")
        plt.ylabel("Difference (mm)")
        plt.tick_params(axis='x', labelsize='small', labelrotation=90)

        plt.subplot(224)
        plt.plot(df_scatter[df_scatter["Metric"]=="CoM_diff2"]["Subject"],
                df_scatter[df_scatter["Metric"]=="CoM_diff2"]["Value"],
                'o', markersize=3)
        plt.title("CoM Z-Continuity: Second-order Diff")
        plt.xlabel("Subjects")
        plt.ylabel("Difference (mm)")
        plt.tick_params(axis='x', labelsize='small', labelrotation=90)
        
        plt.tight_layout()

        if is_png:
            save_results_to = '/space_mounts/data/ACDC/derivatives/plots/plots_sa_segmentation'
            path_name = os.path.join(save_results_to,
            f'Plots_slice_alignment_{df_scatter["Subject"].iloc[0]}_{df_scatter["Subject"].iloc[-1]}.png')
            plt.savefig(path_name)
            plt.close()

    elif is_plot:
         # -----Violin Plots-----
        plt.figure("Slice Boundary Alignment and Contour Metrics", figsize=(12, 10))
        plt.subplot(221)
        sns.violinplot(data=df_violin[df_violin['Metric'] == "Disp_xy"],
                        x='Subject', y="Value")
        plt.ylabel("Distance (mm)")
        plt.title("Displacements between consecutive slices")
        plt.tick_params(axis='x', labelsize='small', labelrotation=90)

        plt.subplot(222)
        sns.violinplot(data=df_violin[df_violin['Metric'] == "Dist_cofmass"],
                        x='Subject', y="Value")
        plt.ylabel("Distance (mm)")
        plt.title("Distances between center of mass and outline")
        plt.tick_params(axis='x', labelsize='small', labelrotation=90)

        plt.subplot(223)
        sns.violinplot(data=df_violin[df_violin['Metric'] == "Contour_Dice"],
                        x='Subject', y="Value")
        plt.ylabel("Dice Coefficient")
        plt.title("Adjacent Slice Contour Overlap (Dice)")
        plt.tick_params(axis='x', labelsize='small', labelrotation=90)

        plt.subplot(224)
        sns.violinplot(data=df_violin[df_violin['Metric'] == "Normal_Similarity"],
                        x='Subject', y="Value")
        plt.ylabel("Cosine Similarity")
        plt.title("Surface Normal Cosine Similarity")
        plt.tick_params(axis='x', labelsize='small', labelrotation=90)
        
        plt.tight_layout()

    # -----90th percentiles------
    if not df_scatter.empty:
        df_90th = df_scatter.groupby(["Subject", "Metric"])["Value"].quantile(0.9)
    else:
        df_90th = pd.Series(dtype=float)
    #print("90th Percentiles:\n",df_90th)

    return df_scatter, df_violin, df_90th


#if __name__ == "__main__":
    #base_path = sys.argv[1]
    #subjects = sys.argv[2]

    # label: 1-> LV; 1,2 -> LV+MYO
    #label = [1]
    #base_path = '/space_mounts/data/ACDC/derivatives/sa_segmentation/'
    #subjects = [i.split("-")[1] for i in os.listdir(base_path) if i.startswith("sub")]

    #visualize_dists(subjects, base_path)
    #compute_displacements_slices(label, subject_test)