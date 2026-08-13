from matplotlib import transforms
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import geoopt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from nimosef.models.nimosef import MultiHeadNetwork

def get_metadata(data_path, mode='train'):
    meta_path = os.path.join(data_path, 'derivatives', 'metadata_participants.tsv')
    manifest_path = os.path.join(data_path, 'derivatives', 'manifests_nimosef', 'dataset_manifest.json')

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    manifest = manifest[mode]
    subjects = [os.path.basename(value).split('_')[0] for value in manifest]
    df_metadata = pd.read_csv(meta_path, sep='\t')
    labels = df_metadata['Group'].unique()
    subject_metadata = df_metadata.set_index('Subject').reindex(subjects)
    subject_metadata = subject_metadata['Group'].to_numpy()
    return labels, subject_metadata, subjects

def knn_accuracy_precomputed(Dmat, y, k=5, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(np.zeros_like(y), y):
        knn = KNeighborsClassifier(n_neighbors=min(k, len(tr)), metric="precomputed")
        knn.fit(Dmat[np.ix_(tr, tr)], y[tr])
        pred = knn.predict(Dmat[np.ix_(te, tr)])
        accs.append(np.mean(pred == y[te]))
    return float(np.mean(accs)), float(np.std(accs))

def evaluate_run(run_path, data_path, device):
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
    
    # Load state dict with strict=False to ignore geoopt internal keys (like isp_c)
    model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    
    shape_codes = model.shape_code.weight.detach().cpu()
    manifold = geoopt.PoincareBall(c=1.0)
    
    norms = shape_codes.norm(dim=-1)
    print(f"  shape code norms: mean={norms.mean():.4f}, max={norms.max():.4f}, "
            f"frac >= 1: {(norms >= 1.0).float().mean():.4f}")
    if norms.max() >= 1.0:
        print("  WARNING: codes violate Poincare ball domain (||x|| >= 1); "
            "manifold.dist() output is not a valid hyperbolic distance for this run.")

    labels, subject_metadata, subjects = get_metadata(data_path)
    labels_to_id = {label: i for i, label in enumerate(labels)}
    color_ids = np.array([labels_to_id[s] for s in subject_metadata if s is not None and not (isinstance(s, float) and np.isnan(s))])
    
    valid_mask = np.array([s is not None and not (isinstance(s, float) and np.isnan(s)) for s in subject_metadata])
    sc_valid = shape_codes[valid_mask]
    y_valid = color_ids
    
    with torch.no_grad():
        D = manifold.dist(sc_valid.unsqueeze(1), sc_valid.unsqueeze(0)).numpy()
    D = np.clip(D, 0, None)
    np.fill_diagonal(D, 0.0)
    
    # Precomputed kNN Accuracy (k=5)
    knn_acc, knn_std = knn_accuracy_precomputed(D, y_valid, k=5)
    
    # Precomputed Silhouette Score
    sil = silhouette_score(D, y_valid, metric="precomputed")
    
    # Permutation test
    rng_perm = np.random.default_rng(0)
    n_perm = 200
    knn_null = np.empty(n_perm)
    for p in range(n_perm):
        y_shuf = rng_perm.permutation(y_valid)
        knn_null[p], _ = knn_accuracy_precomputed(D, y_shuf, k=5, seed=p)
    knn_p = (np.sum(knn_null >= knn_acc) + 1) / (n_perm + 1)
    
    return {
        "knn_acc": knn_acc,
        "knn_std": knn_std,
        "silhouette": sil,
        "p_value": knn_p,
        "num_subjects": len(sc_valid)
    }

def main():
    data_path = "/path/to/ACDC_dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    runs = {
        "Baseline (Adam)": "/path/to/ACDC_dataset/derivatives/nimosef_v2/baseline_m25_sc/experiment_20260331_151125",
        "Baseline (Muon)": "/path/to/ACDC_dataset/derivatives/nimosef_v2/baseline_muon/experiment_20260625_152351",
        "Nimosef (Run 20260619)": "/path/to/ACDC_dataset/derivatives/nimosef_v2/muon_riemann_model/experiment_20260619_160918",
        "Nimosef (Run 20260701)": "/path/to/ACDC_dataset/derivatives/nimosef_v2/muon_riemann_model/experiment_20260701_142855",
    }
    
    print("="*60)
    print("      SHAPE CODE LATENT CLASSIFICATION COMPARISON (k-NN)")
    print("="*60)
    
    results = {}
    for name, run_path in runs.items():
        print(f"Evaluating {name}...")
        try:
            res = evaluate_run(run_path, data_path, device)
            results[name] = res
            print(f"  k-NN Accuracy: {res['knn_acc']:.4f} +/- {res['knn_std']:.4f}")
            print(f"  Silhouette:    {res['silhouette']:.4f}")
            print(f"  p-value (vs chance): {res['p_value']:.4f}")
            print("-" * 60)
        except Exception as e:
            print(f"  Error: {e}")
            print("-" * 60)
            
    # Save results summary to CSV
    df_res = pd.DataFrame(results).T
    df_res.index.name = "Model"
    output_path = "/path/to/ACDC_dataset/derivatives/nimosef_v2/alignment_comparison/latent_classification_results.csv"
    df_res.to_csv(output_path)
    print(f"Summary saved to:\n  {output_path}")

if __name__ == "__main__":
    main()
