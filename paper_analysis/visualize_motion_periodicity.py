import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import geoopt

from nimosef.models.nimosef import MultiHeadNetwork

def compute_motion_trajectory(model, subj, times, hparams, device):
    n = times.shape[0]
    h = model.shape_code(torch.tensor([subj] * n).to(device))
    coords = torch.zeros(n, 3).to(device)
    slice_idx = torch.zeros(n, dtype=torch.long).to(device)
    subj_idx = torch.tensor([subj] * n).to(device)
    cc = model._get_corrected_coords(coords, subj_idx, slice_idx)
    use_trig = hparams.get("use_trig_encoding", True)
    if use_trig:
        inp = model.coord_embedding(cc, times)
    else:
        inp = model.coord_embedding(torch.cat([cc.float(), times.float()], dim=1))
    latent_t = model.latent_projection((inp, h))
    if use_trig:
        st, ct = torch.sin(2 * np.pi * times).float(), torch.cos(2 * np.pi * times).float()
        m = model.motion_head(torch.cat([latent_t, st, ct], dim=1))
    else:
        m = model.motion_head(torch.cat([latent_t, times.float()], dim=1))
    return model.motion_manifold.projx(m)

def load_model_and_get_trajectories(run_path, device):
    model_path = os.path.join(run_path, "checkpoint_last.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(run_path, "model.pth")
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    hparams_path = os.path.join(run_path, "hyperparameters.json")
    if os.path.exists(hparams_path):
        with open(hparams_path, "r") as f:
            hparams = json.load(f)
    else:
        hparams = {"latent_size": 128}

    num_subjects = state_dict["shape_code.weight"].shape[0]
    
    subject_slices = {}
    for key in state_dict:
        if key.startswith("slice_correction.subject_tables."):
            subj_id = key.split(".")[2]
            num_slices = state_dict[key].shape[0]
            subject_slices[int(subj_id)] = num_slices
            
    if not subject_slices and "slice_correction.table.weight" in state_dict:
        total_slices_weights = state_dict["slice_correction.table.weight"].shape[0]
        max_slices = total_slices_weights // num_subjects
        subject_slices = {i: max_slices for i in range(num_subjects)}

    if not subject_slices:
        subject_slices = {i: 10 for i in range(num_subjects)}

    use_trig = hparams.get("use_trig_encoding", True)
    for key in ["latent_projection.net.0.linear.weight", "latent_projection.net.0.freqs.weight"]:
        if key in state_dict:
            cols = state_dict[key].shape[1]
            if cols == 133:
                use_trig = True
            elif cols == 132:
                use_trig = False
            break

    model = MultiHeadNetwork(
        subject_slices=subject_slices,
        num_subjects=num_subjects,
        num_labels=4,
        latent_size=hparams.get("latent_size", 128),
        motion_size=hparams.get("motion_size", 64),
        hidden_size=hparams.get("hidden_size", 128),
        num_res_layers=hparams.get("num_res_layers", 8),
        linear_head=hparams.get("linear_head", True),
        skip_correction=False,
        use_trig_encoding=use_trig
    ).to(device)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    num_time_steps = 50
    times = torch.linspace(0, 1, num_time_steps).view(-1, 1).to(device)
    
    trajectories = []
    with torch.no_grad():
        for subj in range(num_subjects):
            traj = compute_motion_trajectory(model, subj, times, hparams, device).cpu().numpy()
            trajectories.append(traj)
            
    return np.array(trajectories), times.cpu().numpy().flatten()

def get_subject_lv_volumes(file_path):
    import nibabel as nib
    if not os.path.exists(file_path):
        return None
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        zooms = img.header.get_zooms()
        voxel_vol = np.prod(zooms[:3]) / 1000.0 # in ml
        if len(data.shape) == 3:
            volumes = [np.sum(data == 1) * voxel_vol]
        else:
            volumes = []
            for t in range(data.shape[3]):
                volumes.append(np.sum(data[..., t] == 1) * voxel_vol)
        return np.array(volumes)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_subject_bsas():
    metadata_path = "/path/to/ACDC_dataset/derivatives/metadata_participants.tsv"
    if not os.path.exists(metadata_path):
        return {}
    import pandas as pd
    try:
        df = pd.read_csv(metadata_path, sep='\t')
        df['Subject'] = df['Subject'].astype(str)
        bsa_dict = {}
        for idx, row in df.iterrows():
            sub_id = row['Subject']
            h = float(row['Height'])
            w = float(row['Weight'])
            # Mosteller formula
            bsa = np.sqrt(h * w / 3600.0)
            bsa_dict[sub_id] = bsa
        return bsa_dict
    except Exception as e:
        print(f"Error reading BSA metadata: {e}")
        return {}

def load_all_volumes(base_rec_dir, riem_rec_dir):
    import glob
    sub_folders = sorted(glob.glob(os.path.join(riem_rec_dir, "sub-*")))
    
    bsa_dict = load_subject_bsas()
    
    gt_all = []
    base_all = []
    riem_all = []
    es_times = []
    
    print(f"Computing LV volume indices (BSA-normalized) for {len(sub_folders)} subjects...")
    for folder in sub_folders:
        sub_id = os.path.basename(folder)
        gt_path = os.path.join(folder, f"{sub_id}_seg_gt.nii.gz")
        riem_path = os.path.join(folder, f"{sub_id}_seg.nii.gz")
        base_path = os.path.join(base_rec_dir, sub_id, f"{sub_id}_seg.nii.gz")
        
        gt_vols = get_subject_lv_volumes(gt_path)
        riem_vols = get_subject_lv_volumes(riem_path)
        base_vols = get_subject_lv_volumes(base_path)
        
        bsa = bsa_dict.get(sub_id, 1.8)
        
        if gt_vols is not None and len(gt_vols) > 1:
            # Normalize by BSA
            gt_vols = gt_vols / bsa
            if riem_vols is not None and len(riem_vols) > 1:
                riem_vols = riem_vols / bsa
            if base_vols is not None and len(base_vols) > 1:
                base_vols = base_vols / bsa
                
            x_old = np.linspace(0, 1, len(gt_vols))
            x_new = np.linspace(0, 1, 50)
            
            gt_interp = np.interp(x_new, x_old, gt_vols)
            gt_all.append(gt_interp)
            
            # Find ES index
            es_idx = np.argmin(gt_vols)
            es_times.append(es_idx / (len(gt_vols) - 1))
            
            if riem_vols is not None and len(riem_vols) > 1:
                riem_interp = np.interp(x_new, np.linspace(0, 1, len(riem_vols)), riem_vols)
                riem_all.append(riem_interp)
            else:
                riem_all.append(gt_interp)
                
            if base_vols is not None and len(base_vols) > 1:
                base_interp = np.interp(x_new, np.linspace(0, 1, len(base_vols)), base_vols)
                base_all.append(base_interp)
            else:
                base_all.append(gt_interp)
                
    return np.array(gt_all), np.array(base_all), np.array(riem_all), float(np.mean(es_times))

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare high-dimensional motion trajectories.")
    parser.add_argument("--baseline_run", type=str, default="/path/to/ACDC_dataset/derivatives/nimosef_v2/baseline_muon/experiment_20260625_152351",
                        help="Path to baseline model run directory.")
    parser.add_argument("--riemannian_run", type=str, default="/path/to/ACDC_dataset/derivatives/nimosef_v2/muon_riemann_model/experiment_20260701_142855",
                        help="Path to Riemannian model run directory.")
    parser.add_argument("--output_dir", type=str, default="/path/to/ACDC_dataset/derivatives/nimosef_v2/alignment_comparison",
                        help="Output directory for saving plots and JSON metrics.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading baseline model from: {args.baseline_run}")
    base_trajs, times = load_model_and_get_trajectories(args.baseline_run, device)
    
    print(f"Loading Riemannian model from: {args.riemannian_run}")
    riem_trajs, _ = load_model_and_get_trajectories(args.riemannian_run, device)
    
    # Calculate metrics
    def get_metrics_and_curves(trajs):
        loop_closures = []
        path_lengths = []
        curves_ed = []
        curves_mean = []
        
        for traj in trajs:
            norms = np.linalg.norm(traj, axis=1, keepdims=True) + 1e-9
            traj_norm = traj / norms
            
            dist_ed = np.arccos(np.clip(traj_norm @ traj_norm[0], -1.0, 1.0))
            curves_ed.append(dist_ed.tolist())
            
            mean_euclidean = traj_norm.mean(axis=0)
            mean_spherical = mean_euclidean / (np.linalg.norm(mean_euclidean) + 1e-9)
            dist_mean = np.arccos(np.clip(traj_norm @ mean_spherical, -1.0, 1.0))
            curves_mean.append(dist_mean.tolist())
            
            p0 = traj_norm[0]
            p_end = traj_norm[-1]
            closure = np.arccos(np.clip(np.dot(p0, p_end), -1.0, 1.0))
            loop_closures.append(float(closure))
            
            length = 0.0
            n_steps = traj_norm.shape[0]
            for i in range(n_steps - 1):
                length += np.arccos(np.clip(np.dot(traj_norm[i], traj_norm[i+1]), -1.0, 1.0))
            length += np.arccos(np.clip(np.dot(traj_norm[-1], traj_norm[0]), -1.0, 1.0))
            path_lengths.append(float(length))
            
        return {
            "loop_closures": loop_closures,
            "path_lengths": path_lengths,
            "curves_ed": curves_ed,
            "curves_mean": curves_mean,
            "loop_closure_mean": float(np.mean(loop_closures)),
            "loop_closure_std": float(np.std(loop_closures)),
            "path_length_mean": float(np.mean(path_lengths)),
            "path_length_std": float(np.std(path_lengths))
        }
        
    print("Computing motion trajectory metrics...")
    base_metrics = get_metrics_and_curves(base_trajs)
    riem_metrics = get_metrics_and_curves(riem_trajs)
    
    # Compute LV volumes
    base_rec_dir = args.baseline_run.replace("experiment_", "reconstructions_experiment_")
    riem_rec_dir = args.riemannian_run.replace("experiment_", "reconstructions_experiment_")
    gt_vols, base_vols, riem_vols, avg_es_time = load_all_volumes(base_rec_dir, riem_rec_dir)
    print(f"Average End-Systole time relative to cardiac cycle: {avg_es_time:.3f}")
    
    # Save JSON metrics
    metrics_path = os.path.join(args.output_dir, "motion_periodicity_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "baseline": {
                "loop_closure_mean": base_metrics["loop_closure_mean"],
                "loop_closure_std": base_metrics["loop_closure_std"],
                "path_length_mean": base_metrics["path_length_mean"],
                "path_length_std": base_metrics["path_length_std"],
            },
            "riemannian": {
                "loop_closure_mean": riem_metrics["loop_closure_mean"],
                "loop_closure_std": riem_metrics["loop_closure_std"],
                "path_length_mean": riem_metrics["path_length_mean"],
                "path_length_std": riem_metrics["path_length_std"],
            },
            "average_es_time": avg_es_time
        }, f, indent=4)
    print(f"Saved metrics JSON to: {metrics_path}")
    
    # Save 6 individual plots
    base_ed_arr = np.array(base_metrics["curves_ed"])
    base_mean_arr = np.array(base_metrics["curves_mean"])
    riem_ed_arr = np.array(riem_metrics["curves_ed"])
    riem_mean_arr = np.array(riem_metrics["curves_mean"])
    
    base_ed_mean = base_ed_arr.mean(axis=0)
    base_ed_std = base_ed_arr.std(axis=0)
    base_mean_mean = base_mean_arr.mean(axis=0)
    base_mean_std = base_mean_arr.std(axis=0)
    
    riem_ed_mean = riem_ed_arr.mean(axis=0)
    riem_ed_std = riem_ed_arr.std(axis=0)
    riem_mean_mean = riem_mean_arr.mean(axis=0)
    riem_mean_std = riem_mean_arr.std(axis=0)
    
    gt_vol_mean = gt_vols.mean(axis=0)
    gt_vol_std = gt_vols.std(axis=0)
    base_vol_mean = base_vols.mean(axis=0)
    base_vol_std = base_vols.std(axis=0)
    riem_vol_mean = riem_vols.mean(axis=0)
    riem_vol_std = riem_vols.std(axis=0)
    
    # Common plotting parameters
    fig_size = (3.5, 3.2)
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 11,
    })
    
    # ----------------------------------------------------
    # Plot 1: NIMOSEF-E Distance from ED
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(min(20, len(base_trajs))):
        ax.plot(times, base_metrics["curves_ed"][i], linewidth=0.8, linestyle='--', color='gray', alpha=0.3)
    ax.plot(times, base_ed_mean, label="Mean", linewidth=3.5, color='#d62728')
    ax.fill_between(times, base_ed_mean - base_ed_std, base_ed_mean + base_ed_std, color='#d62728', alpha=0.15)
    ax.axvline(x=avg_es_time, color='black', linestyle=':', linewidth=2, label="ES")
    # Row 1: no x-label
    ax.set_ylabel("Geodesic (rad)")
    ax.set_xlabel("Time (frac. cycle)")
    ax.grid(True, alpha=0.3)
    ax.locator_params(nbins=4)
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "nimoefe_ed_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----------------------------------------------------
    # Plot 2: NIMOSEF-E Distance from Mean
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(min(20, len(base_trajs))):
        ax.plot(times, base_metrics["curves_mean"][i], linewidth=0.8, linestyle='--', color='gray', alpha=0.3)
    ax.plot(times, base_mean_mean, label="Mean", linewidth=3.5, color='#d62728')
    ax.fill_between(times, base_mean_mean - base_mean_std, base_mean_mean + base_mean_std, color='#d62728', alpha=0.15)
    ax.axvline(x=avg_es_time, color='black', linestyle=':', linewidth=2, label="ES")
    # Row 1: no x-label, Column 2: no y-label
    ax.set_ylabel("Geodesic (rad)")
    ax.set_xlabel("Time (frac. cycle)")
    ax.grid(True, alpha=0.3)
    ax.locator_params(nbins=4)
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "nimoefe_mean_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----------------------------------------------------
    # Plot 3: NIMOSEF-E Volume Index
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(min(20, len(gt_vols))):
        ax.plot(times, gt_vols[i], linewidth=0.8, linestyle=':', color='gray', alpha=0.2)
        ax.plot(times, base_vols[i], linewidth=0.8, linestyle='--', color='red', alpha=0.15)
    ax.plot(times, gt_vol_mean, label="GT", linewidth=3.5, color='#2ca02c')
    ax.fill_between(times, gt_vol_mean - gt_vol_std, gt_vol_mean + gt_vol_std, color='#2ca02c', alpha=0.15)
    ax.plot(times, base_vol_mean, label="N-E", linewidth=3.5, color='#d62728', linestyle='--')
    ax.fill_between(times, base_vol_mean - base_vol_std, base_vol_mean + base_vol_std, color='#d62728', alpha=0.15)
    ax.axvline(x=avg_es_time, color='black', linestyle=':', linewidth=2, label="ES")
    # Row 1: no x-label, Column 3: yes y-label
    ax.set_ylabel("Volume Index (ml/m²)")
    ax.set_xlabel("Time (frac. cycle)")
    ax.grid(True, alpha=0.3)
    ax.locator_params(nbins=4)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "nimoefe_vol_index.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----------------------------------------------------
    # Plot 4: NIMOSEF-R Distance from ED
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(min(20, len(riem_trajs))):
        ax.plot(times, riem_metrics["curves_ed"][i], linewidth=0.8, color='gray', alpha=0.3)
    ax.plot(times, riem_ed_mean, label="Mean", linewidth=3.5, color='#1f77b4')
    ax.fill_between(times, riem_ed_mean - riem_ed_std, riem_ed_mean + riem_ed_std, color='#1f77b4', alpha=0.15)
    ax.axvline(x=avg_es_time, color='black', linestyle=':', linewidth=2, label="ES")
    ax.set_xlabel("Time (frac. cycle)")
    ax.set_ylabel("Geodesic (rad)")
    ax.grid(True, alpha=0.3)
    ax.locator_params(nbins=4)
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "nimofer_ed_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----------------------------------------------------
    # Plot 5: NIMOSEF-R Distance from Mean
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(min(20, len(riem_trajs))):
        ax.plot(times, riem_metrics["curves_mean"][i], linewidth=0.8, color='gray', alpha=0.3)
    ax.plot(times, riem_mean_mean, label="Mean", linewidth=3.5, color='#1f77b4')
    ax.fill_between(times, riem_mean_mean - riem_mean_std, riem_mean_mean + riem_mean_std, color='#1f77b4', alpha=0.15)
    ax.axvline(x=avg_es_time, color='black', linestyle=':', linewidth=2, label="ES")
    ax.set_xlabel("Time (frac. cycle)")
    ax.set_ylabel("Geodesic (rad)")
    ax.grid(True, alpha=0.3)
    ax.locator_params(nbins=4)
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "nimofer_mean_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----------------------------------------------------
    # Plot 6: NIMOSEF-R Volume Index
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(min(20, len(gt_vols))):
        ax.plot(times, gt_vols[i], linewidth=0.8, linestyle=':', color='gray', alpha=0.2)
        ax.plot(times, riem_vols[i], linewidth=0.8, color='blue', alpha=0.15)
    ax.plot(times, gt_vol_mean, label="GT", linewidth=3.5, color='#2ca02c')
    ax.fill_between(times, gt_vol_mean - gt_vol_std, gt_vol_mean + gt_vol_std, color='#2ca02c', alpha=0.15)
    ax.plot(times, riem_vol_mean, label="N-R", linewidth=3.5, color='#1f77b4')
    ax.fill_between(times, riem_vol_mean - riem_vol_std, riem_vol_mean + riem_vol_std, color='#1f77b4', alpha=0.15)
    ax.axvline(x=avg_es_time, color='black', linestyle=':', linewidth=2, label="ES")
    ax.set_xlabel("Time (frac. cycle)")
    ax.set_ylabel("Volume Index (ml/m²)")
    ax.grid(True, alpha=0.3)
    ax.locator_params(nbins=4)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "nimofer_vol_index.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved all 6 periodicity plots individually to: {args.output_dir}")

if __name__ == "__main__":
    main()
