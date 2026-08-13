import os
import json
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from pytorch3d.ops import knn_points

def compute_class_dice(pred, target, label_id):
    pred_l = (pred == label_id).astype(np.float32)
    target_l = (target == label_id).astype(np.float32)
    intersection = np.sum(pred_l * target_l)
    union = np.sum(pred_l) + np.sum(target_l)
    if np.sum(target_l) == 0 and np.sum(pred_l) == 0:
        return 1.0
    return (2.0 * intersection + 1e-6) / (union + 1e-6)

def compute_all_pc_metrics(pts1, pts2, device):
    # pts1: (1, N, 3), pts2: (1, M, 3)
    pts1_t = torch.tensor(pts1, dtype=torch.float32).unsqueeze(0).to(device)
    pts2_t = torch.tensor(pts2, dtype=torch.float32).unsqueeze(0).to(device)
    
    knn1 = knn_points(pts1_t, pts2_t, K=1, return_nn=False)
    knn2 = knn_points(pts2_t, pts1_t, K=1, return_nn=False)
    
    dists1 = knn1.dists.squeeze(-1).sqrt()  # (1, N)
    dists2 = knn2.dists.squeeze(-1).sqrt()  # (1, M)
    
    # Chamfer
    chamfer = (dists1.mean(dim=1) + dists2.mean(dim=1)).item()
    
    # HF95
    hf95 = torch.max(
        torch.quantile(dists1, 0.95, dim=1),
        torch.quantile(dists2, 0.95, dim=1)
    ).item()
    
    # HF
    hf = torch.max(
        torch.quantile(dists1, 1.0, dim=1),
        torch.quantile(dists2, 1.0, dim=1)
    ).item()
    
    return chamfer, hf95, hf

def process_subject_pc_metrics(reconstructions_folder, subject_id, scale_factor, device):
    pred_boundaries_path = os.path.join(reconstructions_folder, subject_id, f'{subject_id}_pred_boundaries.parquet')
    true_boundaries_path = os.path.join(reconstructions_folder, subject_id, f'{subject_id}_true_boundaries.parquet')
    
    if not (os.path.exists(pred_boundaries_path) and os.path.exists(true_boundaries_path)):
        raise FileNotFoundError(f"Boundary parquets not found for {subject_id}")
        
    pred_boundaries = pd.read_parquet(pred_boundaries_path)
    true_boundaries = pd.read_parquet(true_boundaries_path)
    
    # Scale coordinates
    pred_boundaries['x'] *= scale_factor[0]
    pred_boundaries['y'] *= scale_factor[1]
    pred_boundaries['z'] *= scale_factor[2]
    
    true_boundaries['x'] *= scale_factor[0]
    true_boundaries['y'] *= scale_factor[1]
    true_boundaries['z'] *= scale_factor[2]
    
    # Fast grouping instead of .query inside the loop
    pred_grouped = dict(list(pred_boundaries.groupby('time')))
    true_grouped = dict(list(true_boundaries.groupby('time')))
    
    frames_ids = sorted(list(pred_grouped.keys()))
    
    chamfer_list = []
    hf95_list = []
    hf_list = []
    
    for time_idx in frames_ids:
        pts1 = pred_grouped[time_idx][['x', 'y', 'z']].values
        pts2 = true_grouped[time_idx][['x', 'y', 'z']].values
        
        chamfer_dis, hf95, hf = compute_all_pc_metrics(pts1, pts2, device)
        
        chamfer_list.append(chamfer_dis)
        hf95_list.append(hf95)
        hf_list.append(hf)
        
    return np.mean(chamfer_list), np.mean(hf95_list), np.mean(hf_list)

def main():
    data_path = "/path/to/ACDC_dataset"
    splits_filename = f"{data_path}/derivatives/manifests_nimosef/dataset_manifest.json"
    
    baseline_folder = "/path/to/ACDC_dataset/derivatives/nimosef_v2/baseline_m25_sc/reconstructions_experiment_20260331_151125"
    baseline_muon_folder = "/path/to/ACDC_dataset/derivatives/nimosef_v2/baseline_muon/reconstructions_experiment_20260625_152351"
    # experiment_20260701_142855 is the NIMOSEF-R checkpoint used everywhere
    # else in the paper (Table 1, Table 3, the qualitative figure).
    muon_folder = "/path/to/ACDC_dataset/derivatives/nimosef_v2/muon_riemann_model/reconstructions_experiment_20260701_142855"
    
    output_dir = "/path/to/ACDC_dataset/derivatives/nimosef_v2/alignment_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset manifest
    with open(splits_filename, "r") as f:
        split_data = json.load(f)
        
    bbox = split_data["bbox"]
    scale_factor = np.array(bbox) / 2
    
    subject_ids = []
    for m in ["train", "val", "test"]:
        manifest_paths = split_data[m]
        for p in manifest_paths:
            base = os.path.basename(p)
            subject_id = base.split("_")[0]
            subject_ids.append(subject_id)
    subject_ids = sorted(list(set(subject_ids)))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = []
    
    for idx, subject_id in enumerate(subject_ids):
        # Check if subject folders exist in all three
        if not (os.path.exists(os.path.join(baseline_folder, subject_id)) and 
                os.path.exists(os.path.join(baseline_muon_folder, subject_id)) and
                os.path.exists(os.path.join(muon_folder, subject_id))):
            continue
            
        print(f"[{idx+1}/{len(subject_ids)}] Processing subject {subject_id}...")
        
        try:
            # --- Load segmentations (Using fast np.asanyarray read) ---
            # Baseline (Adam)
            seg_gt_b = np.asanyarray(nib.load(os.path.join(baseline_folder, subject_id, f"{subject_id}_seg_gt.nii.gz")).dataobj)
            seg_pred_b = np.asanyarray(nib.load(os.path.join(baseline_folder, subject_id, f"{subject_id}_seg.nii.gz")).dataobj)
            seg_pred_corr_b = np.asanyarray(nib.load(os.path.join(baseline_folder, subject_id, f"{subject_id}_seg_corrected.nii.gz")).dataobj)
            
            # Baseline (Muon)
            seg_gt_bm = np.asanyarray(nib.load(os.path.join(baseline_muon_folder, subject_id, f"{subject_id}_seg_gt.nii.gz")).dataobj)
            seg_pred_bm = np.asanyarray(nib.load(os.path.join(baseline_muon_folder, subject_id, f"{subject_id}_seg.nii.gz")).dataobj)
            seg_pred_corr_bm = np.asanyarray(nib.load(os.path.join(baseline_muon_folder, subject_id, f"{subject_id}_seg_corrected.nii.gz")).dataobj)
            
            # Nimosef (Muon)
            seg_gt_m = np.asanyarray(nib.load(os.path.join(muon_folder, subject_id, f"{subject_id}_seg_gt.nii.gz")).dataobj)
            seg_pred_m = np.asanyarray(nib.load(os.path.join(muon_folder, subject_id, f"{subject_id}_seg.nii.gz")).dataobj)
            seg_pred_corr_m = np.asanyarray(nib.load(os.path.join(muon_folder, subject_id, f"{subject_id}_seg_corrected.nii.gz")).dataobj)
            
            # --- Dice Scores (using corrected segmentations) ---
            dice_lv_base_corr = compute_class_dice(seg_pred_corr_b, seg_gt_b, 1)
            dice_myo_base_corr = compute_class_dice(seg_pred_corr_b, seg_gt_b, 2)
            dice_rv_base_corr = compute_class_dice(seg_pred_corr_b, seg_gt_b, 3)
            
            dice_lv_basemuon_corr = compute_class_dice(seg_pred_corr_bm, seg_gt_bm, 1)
            dice_myo_basemuon_corr = compute_class_dice(seg_pred_corr_bm, seg_gt_bm, 2)
            dice_rv_basemuon_corr = compute_class_dice(seg_pred_corr_bm, seg_gt_bm, 3)
            
            dice_lv_muon_corr = compute_class_dice(seg_pred_corr_m, seg_gt_m, 1)
            dice_myo_muon_corr = compute_class_dice(seg_pred_corr_m, seg_gt_m, 2)
            dice_rv_muon_corr = compute_class_dice(seg_pred_corr_m, seg_gt_m, 3)
            
            # --- Dice Scores (using uncorrected segmentations) ---
            dice_lv_base_uncorr = compute_class_dice(seg_pred_b, seg_gt_b, 1)
            dice_myo_base_uncorr = compute_class_dice(seg_pred_b, seg_gt_b, 2)
            dice_rv_base_uncorr = compute_class_dice(seg_pred_b, seg_gt_b, 3)
            
            dice_lv_basemuon_uncorr = compute_class_dice(seg_pred_bm, seg_gt_bm, 1)
            dice_myo_basemuon_uncorr = compute_class_dice(seg_pred_bm, seg_gt_bm, 2)
            dice_rv_basemuon_uncorr = compute_class_dice(seg_pred_bm, seg_gt_bm, 3)
            
            dice_lv_muon_uncorr = compute_class_dice(seg_pred_m, seg_gt_m, 1)
            dice_myo_muon_uncorr = compute_class_dice(seg_pred_m, seg_gt_m, 2)
            dice_rv_muon_uncorr = compute_class_dice(seg_pred_m, seg_gt_m, 3)
            
            # --- Point Cloud Metrics ---
            chamfer_base, hf95_base, hf_base = process_subject_pc_metrics(baseline_folder, subject_id, scale_factor, device)
            chamfer_basemuon, hf95_basemuon, hf_basemuon = process_subject_pc_metrics(baseline_muon_folder, subject_id, scale_factor, device)
            chamfer_muon, hf95_muon, hf_muon = process_subject_pc_metrics(muon_folder, subject_id, scale_factor, device)
            
            results.append({
                "Subject": subject_id,
                # Dice Baseline Corrected
                "Base_LV_Dice_Corr": dice_lv_base_corr,
                "Base_MYO_Dice_Corr": dice_myo_base_corr,
                "Base_RV_Dice_Corr": dice_rv_base_corr,
                # Dice Baseline Muon Corrected
                "BaseMuon_LV_Dice_Corr": dice_lv_basemuon_corr,
                "BaseMuon_MYO_Dice_Corr": dice_myo_basemuon_corr,
                "BaseMuon_RV_Dice_Corr": dice_rv_basemuon_corr,
                # Dice Nimosef Corrected
                "Muon_LV_Dice_Corr": dice_lv_muon_corr,
                "Muon_MYO_Dice_Corr": dice_myo_muon_corr,
                "Muon_RV_Dice_Corr": dice_rv_muon_corr,
                # Dice Baseline Uncorrected
                "Base_LV_Dice_Uncorr": dice_lv_base_uncorr,
                "Base_MYO_Dice_Uncorr": dice_myo_base_uncorr,
                "Base_RV_Dice_Uncorr": dice_rv_base_uncorr,
                # Dice Baseline Muon Uncorrected
                "BaseMuon_LV_Dice_Uncorr": dice_lv_basemuon_uncorr,
                "BaseMuon_MYO_Dice_Uncorr": dice_myo_basemuon_uncorr,
                "BaseMuon_RV_Dice_Uncorr": dice_rv_basemuon_uncorr,
                # Dice Nimosef Uncorrected
                "Muon_LV_Dice_Uncorr": dice_lv_muon_uncorr,
                "Muon_MYO_Dice_Uncorr": dice_myo_muon_uncorr,
                "Muon_RV_Dice_Uncorr": dice_rv_muon_uncorr,
                # PC Baseline
                "Base_Chamfer": chamfer_base,
                "Base_HF95": hf95_base,
                "Base_HF": hf_base,
                # PC Baseline Muon
                "BaseMuon_Chamfer": chamfer_basemuon,
                "BaseMuon_HF95": hf95_basemuon,
                "BaseMuon_HF": hf_basemuon,
                # PC Nimosef
                "Muon_Chamfer": chamfer_muon,
                "Muon_HF95": hf95_muon,
                "Muon_HF": hf_muon,
            })
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "baseline_vs_nimosef_comparison_raw.csv"), index=False)
    
    # Statistical calculations
    metrics_to_test = [
        ("LV_Dice_Uncorr", "Base_LV_Dice_Uncorr", "BaseMuon_LV_Dice_Uncorr", "Muon_LV_Dice_Uncorr", True),
        ("MYO_Dice_Uncorr", "Base_MYO_Dice_Uncorr", "BaseMuon_MYO_Dice_Uncorr", "Muon_MYO_Dice_Uncorr", True),
        ("RV_Dice_Uncorr", "Base_RV_Dice_Uncorr", "BaseMuon_RV_Dice_Uncorr", "Muon_RV_Dice_Uncorr", True),
        ("LV_Dice_Corr", "Base_LV_Dice_Corr", "BaseMuon_LV_Dice_Corr", "Muon_LV_Dice_Corr", True),
        ("MYO_Dice_Corr", "Base_MYO_Dice_Corr", "BaseMuon_MYO_Dice_Corr", "Muon_MYO_Dice_Corr", True),
        ("RV_Dice_Corr", "Base_RV_Dice_Corr", "BaseMuon_RV_Dice_Corr", "Muon_RV_Dice_Corr", True),
        ("Chamfer", "Base_Chamfer", "BaseMuon_Chamfer", "Muon_Chamfer", False),
        ("HF95", "Base_HF95", "BaseMuon_HF95", "Muon_HF95", False),
        ("HF", "Base_HF", "BaseMuon_HF", "Muon_HF", False)
    ]
    
    stats_list = []
    
    p_vals_vs_base = []
    p_vals_vs_basemuon = []
    
    for name, base_col, basemuon_col, muon_col, higher_is_better in metrics_to_test:
        base_vals = df[base_col].values
        basemuon_vals = df[basemuon_col].values
        muon_vals = df[muon_col].values
        
        _, p_vs_base = wilcoxon(muon_vals, base_vals)
        _, p_vs_basemuon = wilcoxon(muon_vals, basemuon_vals)
        
        p_vals_vs_base.append(p_vs_base)
        p_vals_vs_basemuon.append(p_vs_basemuon)
        
        stats_list.append({
            "Metric": name,
            "Base_Median": np.median(base_vals),
            "Base_IQR": np.percentile(base_vals, [25, 75]),
            "BaseMuon_Median": np.median(basemuon_vals),
            "BaseMuon_IQR": np.percentile(basemuon_vals, [25, 75]),
            "Muon_Median": np.median(muon_vals),
            "Muon_IQR": np.percentile(muon_vals, [25, 75]),
            "Raw_P_vs_Base": p_vs_base,
            "Raw_P_vs_BaseMuon": p_vs_basemuon,
            "HigherIsBetter": higher_is_better
        })
        
    all_p_vals = p_vals_vs_base + p_vals_vs_basemuon
    _, all_p_adj, _, _ = multipletests(all_p_vals, alpha=0.05, method="fdr_bh")
    
    p_adj_vs_base = all_p_adj[:9]
    p_adj_vs_basemuon = all_p_adj[9:]
    
    for i, item in enumerate(stats_list):
        item["FDR_P_vs_Base"] = p_adj_vs_base[i]
        item["FDR_P_vs_BaseMuon"] = p_adj_vs_basemuon[i]
        
        p1 = p_adj_vs_base[i]
        p2 = p_adj_vs_basemuon[i]
        
        item["Sig_vs_Base"] = "p < 0.001" if p1 < 0.001 else ("p < 0.01" if p1 < 0.01 else ("p < 0.05" if p1 < 0.05 else "ns"))
        item["Sig_vs_BaseMuon"] = "p < 0.001" if p2 < 0.001 else ("p < 0.01" if p2 < 0.01 else ("p < 0.05" if p2 < 0.05 else "ns"))
        
    df_stats = pd.DataFrame(stats_list)
    df_stats.to_csv(os.path.join(output_dir, "baseline_vs_nimosef_comparison_summary.csv"), index=False)
    
    print("\n" + "="*70)
    print("      COMPARATIVE ANALYSIS: BASELINE (ADAM) vs BASELINE (MUON) vs NIMOSEF")
    print("="*70)
    
    for idx, row in df_stats.iterrows():
        name = row["Metric"]
        b_med, b_iqr = row["Base_Median"], row["Base_IQR"]
        bm_med, bm_iqr = row["BaseMuon_Median"], row["BaseMuon_IQR"]
        m_med, m_iqr = row["Muon_Median"], row["Muon_IQR"]
        unit = " mm" if name in ["Chamfer", "HF95", "HF"] else ""
        
        print(f"{name}:")
        print(f"  Baseline (Adam): {b_med:.4f} [{b_iqr[0]:.4f}, {b_iqr[1]:.4f}]{unit}")
        print(f"  Baseline (Muon): {bm_med:.4f} [{bm_iqr[0]:.4f}, {bm_iqr[1]:.4f}]{unit}")
        print(f"  Nimosef (Muon):  {m_med:.4f} [{m_iqr[0]:.4f}, {m_iqr[1]:.4f}]{unit}")
        print(f"  Wilcoxon p (vs Base Adam): {row['FDR_P_vs_Base']:.3e} ({row['Sig_vs_Base']})")
        print(f"  Wilcoxon p (vs Base Muon): {row['FDR_P_vs_BaseMuon']:.3e} ({row['Sig_vs_BaseMuon']})")
        print("-" * 70)
        
    # Generate LaTeX Table code
    latex_code = """\\begin{table}[t]
\\centering
\\caption{Comparative analysis of segmentation quality (Dice) and point cloud boundary errors (Chamfer, HF95) between the baseline configurations and Nimosef.}
\\label{tab:baseline_vs_nimosef}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Metric} & \\textbf{Baseline (Adam)} & \\textbf{Baseline (Muon)} & \\textbf{Nimosef} & \\textbf{$p$ (vs Base Adam)} & \\textbf{$p$ (vs Base Muon)} \\\\
\\midrule
\\multicolumn{6}{l}{\\textit{Uncorrected Segmentations (vs. nnU-Net GT)}} \\\\
"""
    corr_added = False
    pc_added = False
    
    for idx, row in df_stats.iterrows():
        name = row["Metric"]
        b_med, b_iqr = row["Base_Median"], row["Base_IQR"]
        bm_med, bm_iqr = row["BaseMuon_Median"], row["BaseMuon_IQR"]
        m_med, m_iqr = row["Muon_Median"], row["Muon_IQR"]
        higher_is_better = row["HigherIsBetter"]
        
        if "Corr" in name and not corr_added:
            latex_code += "\\midrule\n\\multicolumn{6}{l}{\\textit{Slice-Corrected (SC) Segmentations (vs. nnU-Net GT)}} \\\\\n"
            corr_added = True
        elif name in ["Chamfer", "HF95", "HF"] and not pc_added:
            latex_code += "\\midrule\n\\multicolumn{6}{l}{\\textit{Uncorrected Boundary Point Cloud Errors (vs. nnU-Net GT boundaries)}} \\\\\n"
            pc_added = True
            
        vals = [b_med, bm_med, m_med]
        if higher_is_better:
            best_idx = np.argmax(vals)
        else:
            best_idx = np.argmin(vals)
            
        def fmt(val, iqr, is_best):
            s = f"{val:.3f} [{iqr[0]:.3f}, {iqr[1]:.3f}]"
            return f"\\mathbf{{{s}}}" if is_best else s
            
        b_str = "$" + fmt(b_med, b_iqr, best_idx == 0) + "$"
        bm_str = "$" + fmt(bm_med, bm_iqr, best_idx == 1) + "$"
        m_str = "$" + fmt(m_med, m_iqr, best_idx == 2) + "$"
        
        p1_str = "p < 0.001" if row["FDR_P_vs_Base"] < 0.001 else f"p = {row['FDR_P_vs_Base']:.3f}"
        p2_str = "p < 0.001" if row["FDR_P_vs_BaseMuon"] < 0.001 else f"p = {row['FDR_P_vs_BaseMuon']:.3f}"
        
        if name == "LV_Dice_Uncorr":
            metric_label = "~~LV Dice"
        elif name == "MYO_Dice_Uncorr":
            metric_label = "~~MYO Dice"
        elif name == "RV_Dice_Uncorr":
            metric_label = "~~RV Dice"
        elif name == "LV_Dice_Corr":
            metric_label = "~~LV Dice"
        elif name == "MYO_Dice_Corr":
            metric_label = "~~MYO Dice"
        elif name == "RV_Dice_Corr":
            metric_label = "~~RV Dice"
        elif name == "HF95":
            metric_label = "~~HF95 (mm)"
        elif name == "HF":
            metric_label = "~~HF (mm)"
        elif name == "Chamfer":
            metric_label = "~~Chamfer Distance (mm)"
        else:
            metric_label = name.replace("_", " ")
            
        latex_code += f"{metric_label} & {b_str} & {bm_str} & {m_str} & {p1_str} & {p2_str} \\\\\n"
        
    latex_code += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    latex_path = os.path.join(output_dir, "baseline_vs_nimosef_comparison_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_code)
        
    print(f"LaTeX comparison table code saved to:\n  {latex_path}")

if __name__ == "__main__":
    main()
