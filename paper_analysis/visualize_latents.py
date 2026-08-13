import os
import sys
import glob
import json
import torch
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D projection
from sklearn.decomposition import PCA
import geoopt

from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import umap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nimosef.models.nimosef import MultiHeadNetwork

# ========================================================
# 3D SPHERE VIEWING ANGLE CONFIGURATION
# ========================================================
# Set to None to automatically orient the camera so the
# average cardiac loop faces the viewer (recommended).
# Override with integer degrees to lock a specific angle.
ELEVATION = None  # e.g. 30   to fix elevation
AZIMUTH   = None  # e.g. -60  to fix azimuth
# ========================================================

def get_metadata(data_path, mode='train'):
    meta_path = os.path.join(data_path, 'derivatives', 'metadata_participants.tsv')
    manifest_path = os.path.join(data_path, 'derivatives', 'manifests_nimosef', 'dataset_manifest.json')

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    manifest = manifest[mode]
    subjects = [os.path.basename(value).split('_')[0] for value in manifest]
    df_metadata = pd.read_csv(meta_path, sep='\t')
    print(df_metadata.head())

    labels = df_metadata['Group'].unique()
    subject_metadata = df_metadata.set_index('Subject').reindex(subjects)
    subject_metadata = subject_metadata['Group'].to_numpy()

    return labels, subject_metadata

def plot_poincare_disk(ax):
    """Draws the boundary of the Poincaré disk."""
    circle = plt.Circle((0, 0), 1, color='none', ec='black', linestyle='--')
    ax.add_artist(circle)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title("Shape Codes (Poincaré Disk via PCA in Tangent Space)")
    ax.axis('off')

def compute_frechet_mean(xs, manifold, lr=0.01, steps=100):
    # Initialize at the Einstein midpoint as a good starting guess
    with torch.no_grad():
        y_init = geoopt.manifolds.stereographic.math.weighted_midpoint(xs, k=-manifold.c, reducedim=[0])
    y = geoopt.ManifoldParameter(y_init.clone(), manifold=manifold)
    optimizer = geoopt.optim.RiemannianAdam([y], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = manifold.dist(xs, y).pow(2).mean()
        loss.backward()
        optimizer.step()
    return y.detach()

def compute_karcher_mean_sphere(xs, manifold, lr=0.05, steps=100):
    """Intrinsic (Karcher) mean of points on a sphere via Riemannian optimization.
    xs: (N, D) tensor of points already on the manifold.
    Mirrors compute_frechet_mean but on the sphere instead of the Poincare ball.
    """
    with torch.no_grad():
        y_init = torch.nn.functional.normalize(xs.mean(dim=0), p=2, dim=-1)
    y = geoopt.ManifoldParameter(y_init.clone(), manifold=manifold)
    optimizer = geoopt.optim.RiemannianAdam([y], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = manifold.dist(xs, y.unsqueeze(0)).pow(2).mean()
        loss.backward()
        optimizer.step()
    return y.detach()

def compute_motion_trajectory(model, subj, times, hparams, device):
    """Forward pass at the center coordinate to extract a subject's motion codes
    over the cardiac phase. Returns (num_time_steps, motion_size) on the manifold."""
    num_time_steps = times.shape[0]
    h = model.shape_code(torch.tensor([subj] * num_time_steps).to(device))
    coords = torch.zeros(num_time_steps, 3).to(device)
    slice_indices = torch.zeros(num_time_steps, dtype=torch.long).to(device)
    subj_indices = torch.tensor([subj] * num_time_steps).to(device)
    corrected_coords = model._get_corrected_coords(coords, subj_indices, slice_indices)

    use_trig_encoding = hparams.get("use_trig_encoding", True)
    if use_trig_encoding:
        input_data = model.coord_embedding(corrected_coords, times)
    else:
        input_data = model.coord_embedding(
            torch.cat([corrected_coords.float(), times.float()], dim=1))

    latent_t = model.latent_projection((input_data, h))

    if use_trig_encoding:
        sin_t = torch.sin(2 * math.pi * times).float()
        cos_t = torch.cos(2 * math.pi * times).float()
        motion_code = model.motion_head(torch.cat([latent_t, sin_t, cos_t], dim=1))
    else:
        motion_code = model.motion_head(torch.cat([latent_t, times.float()], dim=1))
    return model.motion_manifold.projx(motion_code)

def slerp(p0, p1, num_interp=10):
    """Spherical Linear Interpolation between two points on S^2."""
    dot = np.dot(p0, p1)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    if np.abs(theta) < 1e-6:
        return np.linspace(p0, p1, num_interp)
    t = np.linspace(0, 1, num_interp)[:, np.newaxis]
    sin_theta = np.sin(theta)
    val = np.sin((1 - t) * theta) / sin_theta * p0 + np.sin(t * theta) / sin_theta * p1
    return val

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default='/path/to/ACDC_dataset/derivatives/nimosef_v2/muon_riemann_model/experiment_20260701_142855')
    args = ap.parse_args()
    
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
    })

    data_path = "/path/to/ACDC_dataset"
    latest_run = args.run_dir

    # Get metadata
    labels, subject_metadata = get_metadata(data_path)
    labels_to_id = {label: i for i, label in enumerate(labels)}
    color_ids = np.array([labels_to_id[subject] for subject in subject_metadata])

    model_path = os.path.join(latest_run, "checkpoint_last.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(latest_run, "model.pth")
        if not os.path.exists(model_path):
            print(f"No model weights found in {latest_run}")
            return
            
    print(f"Loading model from {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Load hyperparameters
    hparams_path = os.path.join(latest_run, "hyperparameters.json")
    if os.path.exists(hparams_path):
        with open(hparams_path, "r") as f:
            hparams = json.load(f)
    else:
        hparams = {
            "latent_size": 128,
            "motion_size": 64,
            "hidden_size": 128,
            "num_res_layers": 8,
            "linear_head": True,
            "use_trig_encoding": True
        }

    # Extract sizes from state_dict to confirm
    num_subjects = state_dict["shape_code.weight"].shape[0]
    
    # Determine subject slices from the trained state dict
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
        
    # Robust trig encoding inference from shape
    use_trig = hparams.get("use_trig_encoding", True)
    for key in ["latent_projection.net.0.linear.weight", "latent_projection.net.0.freqs.weight"]:
        if key in state_dict:
            cols = state_dict[key].shape[1]
            if cols == 133:
                use_trig = True
            elif cols == 132:
                use_trig = False
            break
        
    # Reconstruct the model architecture
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
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # Filter out expected slice correction helper/weight variations if needed
    important_missing = [k for k in missing_keys if not k.startswith("slice_correction")]
    if important_missing:
        print("Warning: Missing keys in state dict:", important_missing)
    if unexpected_keys:
        print("Warning: Unexpected keys in state dict:", unexpected_keys)
    model.eval()    # ---------------------------------------------------------
    # 1. Visualize Shape Codes (Poincaré Ball)
    # --- Mathematical Formulation ---
    # Principal Geodesic Analysis (PGA) for projecting manifold-valued data:
    # 1. Map data to the tangent space at the origin T_0(M) via the logarithmic map.
    #    T_0(M) is a Euclidean vector space where linear projections (PCA) are valid.
    # 2. Perform Euclidean PCA to obtain a 2D tangent subspace.
    # 3. Map the 2D tangent coordinates back to the Poincare disk via the exponential map.
    #
    # --- On the radial rescaling ---
    # PCA from 128D to 2D discards variance, so the retained tangent vectors are
    # shorter than the originals and expmap0 contracts them toward the disk center.
    # We apply a single positive radial scale in the 2D tangent plane purely for
    # visibility. This preserves directions, collinearity, and distance RATIOS in
    # the tangent plane, but note expmap0 is nonlinear in radius: the rescaling is
    # a deliberate visualization choice, not a distance-preserving (isometric) map.
    #
    # --- Consistency of the average ---
    # The Frechet mean is computed on the full manifold, then mapped through the
    # SAME logmap0 -> PCA -> rescale -> expmap0 pipeline as the data. We also report
    # its residual to the PCA plane so an off-plane mean is not silently trusted.
    #
    # --- References ---
    # [1] Fletcher, P. T., et al. (2004). "Principal geodesic analysis for the study of
    #     nonlinear statistics of shape." IEEE TMI, 23(8), 995-1005.
    # [2] Nickel, M., & Kiela, D. (2017). "Poincaré embeddings for learning
    #     hierarchical representations." NeurIPS 2017.
    # [3] Ganea, O. E., Bécigneul, G., & Hofmann, T. (2018). "Hyperbolic Neural
    #     Networks." NeurIPS 2018.
    # ---------------------------------------------------------
    EPS = 1e-9
    shape_codes = model.shape_code.weight.detach().cpu()
    manifold = geoopt.PoincareBall(c=1.0)

    # Map data to tangent space at origin, then PCA to 2D
    tangent_codes = manifold.logmap0(shape_codes).numpy()
    pca = PCA(n_components=2)
    tangent_2d = pca.fit_transform(tangent_codes)

    # Single radial rescale in the 2D tangent plane (visibility only)
    tangent_2d_norms = np.linalg.norm(tangent_2d, axis=1, keepdims=True)
    max_tangent_2d = float(np.max(tangent_2d_norms))
    target_max_norm = 2.5  # ~0.85 Euclidean radius on the disk after expmap0
    scale = (target_max_norm / max_tangent_2d) if max_tangent_2d > EPS else 1.0
    tangent_2d = tangent_2d * scale

    # Map back to the 2D Poincare disk
    manifold_2d = geoopt.PoincareBall(c=1.0)
    poincare_2d = manifold_2d.expmap0(torch.tensor(tangent_2d)).numpy()

    fig = plt.figure(figsize=(15, 6.5))
    cmap = plt.get_cmap('tab20', len(labels))

    # --- Average shape: true Frechet mean, mapped through the same pipeline ---
    avg_shape_code = compute_frechet_mean(shape_codes.to(device), manifold).cpu()
    avg_tangent = manifold.logmap0(avg_shape_code.unsqueeze(0)).numpy()  # (1, 128)

    # Project onto the fitted PCA plane, then reconstruct to measure off-plane residual.
    # pca.transform centers with the same mean used during fit, so the residual is exact.
    avg_tangent_2d = pca.transform(avg_tangent)                    # (1, 2)
    avg_reconstructed = pca.inverse_transform(avg_tangent_2d)      # (1, 128)
    residual = float(np.linalg.norm(avg_tangent - avg_reconstructed))
    total_norm = float(np.linalg.norm(avg_tangent - pca.mean_))
    rel_residual = residual / (total_norm + EPS)
    print(f"Frechet mean off-plane residual: {residual:.4f} "
          f"({100 * rel_residual:.1f}% of its centered norm)")
    if rel_residual > 0.5:
        print("  WARNING: mean lies largely outside the 2D PCA plane; "
              "its plotted position is only a rough indicator.")

    # Apply the SAME radial rescale, then expmap0
    avg_tangent_2d = avg_tangent_2d * scale
    avg_poincare_2d = manifold_2d.expmap0(torch.tensor(avg_tangent_2d)).numpy()

    # ---------------------------------------------------------
    # 1b. Quantify shape-code class separation in the FULL space
    # --- The 2D disk is a lossy projection (~33% variance). To judge whether the
    #     diagnostic groups separate at all, we measure separation on the full
    #     hyperbolic metric, not the projection, and compare against a label-
    #     shuffled baseline so we know what chance looks like for these class sizes.
    # ---------------------------------------------------------
    # Drop any subjects without a metadata label so metrics are well-defined
    valid_mask = np.array([s is not None and not (isinstance(s, float) and np.isnan(s))
                           for s in subject_metadata])
    sc_valid = shape_codes[valid_mask]
    y_valid = color_ids[valid_mask]
    n_valid = int(valid_mask.sum())

    # Full pairwise hyperbolic distance matrix (N, N) on the Poincare ball
    with torch.no_grad():
        D = manifold.dist(sc_valid.unsqueeze(1), sc_valid.unsqueeze(0)).numpy()
    D = np.clip(D, 0, None)
    np.fill_diagonal(D, 0.0)

    # --- Silhouette on the precomputed hyperbolic distances ---
    # Needs at least 2 classes and each class with >= 2 members
    unique, counts = np.unique(y_valid, return_counts=True)
    can_silhouette = len(unique) >= 2 and np.all(counts >= 2)
    if can_silhouette:
        sil = silhouette_score(D, y_valid, metric="precomputed")
    else:
        sil = float("nan")

    # --- k-NN classification accuracy via precomputed distances ---
    # Leave-one-out style through stratified CV; distances are looked up, not refit.
    def knn_accuracy_precomputed(Dmat, y, k=5, n_splits=5, seed=0):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs = []
        for tr, te in skf.split(np.zeros_like(y), y):
            # restrict columns to training points so test never sees its own neighbors
            knn = KNeighborsClassifier(n_neighbors=min(k, len(tr)), metric="precomputed")
            knn.fit(Dmat[np.ix_(tr, tr)], y[tr])
            pred = knn.predict(Dmat[np.ix_(te, tr)])
            accs.append(np.mean(pred == y[te]))
        return float(np.mean(accs)), float(np.std(accs))

    knn_acc, knn_std = knn_accuracy_precomputed(D, y_valid, k=5)

    # --- Shuffled baseline: same metrics with labels permuted ---
    rng_perm = np.random.default_rng(0)
    n_perm = 200
    sil_null = np.full(n_perm, np.nan)
    knn_null = np.empty(n_perm)
    for p in range(n_perm):
        y_shuf = rng_perm.permutation(y_valid)
        if can_silhouette:
            sil_null[p] = silhouette_score(D, y_shuf, metric="precomputed")
        knn_null[p], _ = knn_accuracy_precomputed(D, y_shuf, k=5, seed=p)

    # Empirical p-values (one-sided: is the true value better than chance?)
    knn_p = (np.sum(knn_null >= knn_acc) + 1) / (n_perm + 1)
    sil_p = ((np.nansum(sil_null >= sil) + 1) / (n_perm + 1)) if can_silhouette else float("nan")

    chance = 1.0 / len(unique)
    print("\n--- Shape-code separation (full hyperbolic space) ---")
    print(f"  N = {n_valid} subjects, {len(unique)} classes, chance acc = {chance:.3f}")
    print(f"  k-NN accuracy : {knn_acc:.3f} +/- {knn_std:.3f}")
    print(f"    shuffled    : {knn_null.mean():.3f} +/- {knn_null.std():.3f}  (p = {knn_p:.4f})")
    print(f"  silhouette    : {sil:.3f}")
    print(f"    shuffled    : {np.nanmean(sil_null):.3f} +/- {np.nanstd(sil_null):.3f}  (p = {sil_p:.4f})")
    print("  Interpretation: p < 0.05 on k-NN means groups ARE separable in latent")
    print("  space even if the 2D disk looks intermingled. p ~ chance means the")
    print("  shape codes capture anatomy shared across diagnoses (still a finding).")

    # ---------------------------------------------------------
    # 1c. UMAP of shape codes on the hyperbolic metric (unsupervised)
    # --- Trust this ONLY if it agrees with the k-NN number above. UMAP can
    #     manufacture apparent clusters from noise, so it is a visual companion
    #     to the metric, not a substitute for it.
    # ---------------------------------------------------------
    ax_umap = fig.add_subplot(121)
    if n_valid >= 5:
        reducer = umap.UMAP(
            n_components=2,
            metric="precomputed",
            n_neighbors=min(15, n_valid - 1),
            min_dist=0.1,
            random_state=0,
        )
        emb = reducer.fit_transform(D)
        sc_umap = ax_umap.scatter(emb[:, 0], emb[:, 1], c=y_valid, cmap=cmap,
                                  s=80, alpha=0.85, edgecolors='w')
        ax_umap.set_title("Shape Codes (UMAP on hyperbolic metric)")
        ax_umap.set_xticks([]); ax_umap.set_yticks([])
        cbar_u = fig.colorbar(sc_umap, ax=ax_umap, label='Diagnostic Group', shrink=0.5)
        cbar_u.set_ticks(np.arange(len(labels)))
        cbar_u.set_ticklabels(labels)
    else:
        ax_umap.text(0.5, 0.5, "Too few subjects for UMAP", ha='center', va='center')
        ax_umap.axis('off')

    # ---------------------------------------------------------
    # 2. Visualize Motion Codes (Sphere)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 2. Visualize Motion Codes (Sphere)
    # --- The average trajectory is the INTRINSIC (Karcher) mean over ALL subjects,
    #     computed per time step on the sphere. This mirrors the Riemannian Frechet
    #     mean used for the shape codes, so both panels use consistent statistics.
    #     We plot only a few individual subjects for visual texture; the average is
    #     independent of that subsample and is therefore deterministic.
    # ---------------------------------------------------------
    num_time_steps = 30
    times = torch.linspace(0, 1, num_time_steps).view(-1, 1).to(device)

    # --- Build the population average over ALL subjects ---
    # all_subject_motions: (num_subjects, num_time_steps, motion_size)
    with torch.no_grad():
        all_subject_motions = torch.stack([
            compute_motion_trajectory(model, subj, times, hparams, device).cpu()
            for subj in range(num_subjects)
        ], dim=0)

    motion_manifold = model.motion_manifold
    # Intrinsic Karcher mean per time step across all subjects
    avg_motion_trajectory = torch.stack([
        compute_karcher_mean_sphere(all_subject_motions[:, t, :], motion_manifold)
        for t in range(num_time_steps)
    ], dim=0).numpy()  # (num_time_steps, motion_size)

    # --- Pick a few subjects to display individually (decoupled from the average) ---
    rng = np.random.default_rng(0)  # seeded for reproducible figures
    sample_subjects = rng.choice(num_subjects, size=min(5, num_subjects), replace=False)

    all_motions = []
    subject_labels = []
    time_labels = []
    for subj in sample_subjects:
        all_motions.append(all_subject_motions[subj].numpy())
        subject_labels.extend([subj] * num_time_steps)
        time_labels.extend(np.linspace(0, 1, num_time_steps))
    all_motions = np.concatenate(all_motions, axis=0)
    # Fit on the FULL population so the coordinate system is globally consistent.
    motion_size = hparams.get("motion_size", 64)
    pca_sphere = PCA(n_components=3)
    pop_motions = all_subject_motions.numpy().reshape(-1, motion_size)
    pca_sphere.fit(pop_motions)

    motion_3d = pca_sphere.transform(all_motions)
    avg_motion_3d = pca_sphere.transform(avg_motion_trajectory)

    # Project each point to the unit sphere — standard visualisation of trajectories on S^2
    motion_3d     = motion_3d     / (np.linalg.norm(motion_3d,     axis=1, keepdims=True) + 1e-9)
    avg_motion_3d = avg_motion_3d / (np.linalg.norm(avg_motion_3d, axis=1, keepdims=True) + 1e-9)

    # --- Auto-compute best viewing angle so the average loop faces the camera ---
    # Use the raw (pre-sphere-normalization) centroid of the average trajectory in PCA space.
    # Normalizing each point maps them to S^2, but the loop's centroid in the
    # original PCA space points toward the "center of mass" of the loop's region on
    # the sphere — looking from that direction shows the loop spread around you.
    avg_motion_raw = pca_sphere.transform(avg_motion_trajectory)   # un-normalized, (T, 3)
    centroid = avg_motion_raw.mean(axis=0)                         # (3,)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
    # Flip so the camera is always above the equator (positive elevation)
    if centroid_norm[2] < 0:
        centroid_norm = -centroid_norm
    auto_elev = float(np.degrees(np.arcsin(np.clip(centroid_norm[2], -1, 1))))
    auto_azim = float(np.degrees(np.arctan2(centroid_norm[1], centroid_norm[0])))
    print(f"Auto-computed view: elev={auto_elev:.1f}°, azim={auto_azim:.1f}°")

    ax2 = fig.add_subplot(122, projection='3d')
    # Use manual override if set, otherwise use the auto-computed angle
    _use_elev = ELEVATION if (ELEVATION is not None) else auto_elev
    _use_azim = AZIMUTH if (AZIMUTH is not None) else auto_azim
    ax2.view_init(elev=_use_elev, azim=_use_azim)
    ax2.set_title("Motion Trajectories (Spherical Space via PCA)")

    # Faint wireframe sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax2.plot_wireframe(x, y, z, alpha=0.15)

    # Custom non-red premium color palette for sample subjects (steel blue, forest green, purple, slate, teal)
    colors = ['#2b5c8f', '#2a7c4e', '#6f4a8a', '#8b7a5d', '#4a7c8a']
    for i, subj in enumerate(sample_subjects):
        mask = (np.array(subject_labels) == subj)
        pts = motion_3d[mask]
        ts = np.array(time_labels)[mask]

        # Slerp-interpolate adjacent points for a smooth geodesic line, including closing the loop
        interp_pts = []
        for idx in range(len(pts) - 1):
            interp_pts.append(slerp(pts[idx], pts[idx+1], num_interp=10)[:-1])
        interp_pts.append(slerp(pts[-1], pts[0], num_interp=10))
        pts_slerped = np.concatenate(interp_pts, axis=0)

        # Plot individual subject trajectories as dashed lines and add a single legend entry
        label = 'Individual Trajectories (Samples)' if i == 0 else None
        ax2.plot(pts_slerped[:, 0], pts_slerped[:, 1], pts_slerped[:, 2],
                 color=colors[i % len(colors)], alpha=0.45, linewidth=2.5, linestyle='--', label=label)
        sc = ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=ts, cmap='cool',
                         vmin=0, vmax=1, s=60, alpha=0.70, edgecolor='none', zorder=4)
        ax2.set_facecolor('white')

    # Average motion trajectory (intrinsic mean over all subjects), slerp-smoothed
    avg_interp_pts = []
    for idx in range(len(avg_motion_3d) - 1):
        avg_interp_pts.append(slerp(avg_motion_3d[idx], avg_motion_3d[idx+1], num_interp=10)[:-1])
    avg_interp_pts.append(slerp(avg_motion_3d[-1], avg_motion_3d[0], num_interp=10))
    avg_pts_slerped = np.concatenate(avg_interp_pts, axis=0)

    ax2.plot(avg_pts_slerped[:, 0], avg_pts_slerped[:, 1], avg_pts_slerped[:, 2],
             color='red', linewidth=4, zorder=6, label='Average Motion (all subjects)')
    ax2.scatter(avg_motion_3d[0, 0], avg_motion_3d[0, 1], avg_motion_3d[0, 2],
                color='red', marker='s', s=100, edgecolors='black', zorder=7, label='Start (t=0)')

    fig.colorbar(sc, ax=ax2, label='Cardiac Cycle (0 -> 1)', shrink=0.5, pad=0.1)
    ax2.legend()

    ax2.set_xlim([-1.1, 1.1])
    ax2.set_ylim([-1.1, 1.1])
    ax2.set_zlim([-1.1, 1.1])
    ax2.axis('off')
 
    out_path = os.path.join(latest_run, "latent_space_visualization_new.png")
    # plt.tight_layout()
    fig.subplots_adjust(left=0.04, right=0.97, wspace=0.30)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"Success! Visualization saved to: {out_path}")
    
    # Save copy to alignment_comparison directory with a run-specific name
    run_name = os.path.basename(os.path.normpath(latest_run))
    parent_name = os.path.basename(os.path.dirname(os.path.normpath(latest_run)))
    comparison_filename = f"latent_space_visualization_{parent_name}_{run_name}.png"
    comparison_path = os.path.join("/path/to/ACDC_dataset/derivatives/nimosef_v2/alignment_comparison", comparison_filename)
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"Copied visualization to: {comparison_path}")

    # ---------------------------------------------------------
    # Save Separate Figure 1: Shape UMAP
    # ---------------------------------------------------------
    fig_shape, ax_shape = plt.subplots(figsize=(4.5, 4.0))
    if n_valid >= 5:
        sc_umap_ind = ax_shape.scatter(emb[:, 0], emb[:, 1], c=y_valid, cmap=cmap,
                                       s=80, alpha=0.85, edgecolors='w')
        ax_shape.set_xticks([]); ax_shape.set_yticks([])
        cbar_u_ind = fig_shape.colorbar(sc_umap_ind, ax=ax_shape, orientation='horizontal', pad=0.05, shrink=0.95)
        cbar_u_ind.set_ticks(np.arange(len(labels)))
        cbar_u_ind.set_ticklabels(labels)
    else:
        ax_shape.text(0.5, 0.5, "Too few subjects for UMAP", ha='center', va='center')
        ax_shape.axis('off')
    
    shape_filename = f"shape_codes_{parent_name}_{run_name}.png"
    shape_path = os.path.join("/path/to/ACDC_dataset/derivatives/nimosef_v2/alignment_comparison", shape_filename)
    fig_shape.tight_layout()
    fig_shape.savefig(shape_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig_shape)
    print(f"Saved separate shape UMAP to: {shape_path}")

    # Trim shape UMAP image whitespace and copy to paper
    try:
        from PIL import Image
        import shutil
        img = Image.open(shape_path)
        arr = np.array(img)
        mask = np.any(arr < 250, axis=-1)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if np.any(rows) and np.any(cols):
            ymin, ymax = np.where(rows)[0][0], np.where(rows)[0][-1]
            xmin, xmax = np.where(cols)[0][0], np.where(cols)[0][-1]
            pad = 10
            h, w, _ = arr.shape
            ymin = max(0, ymin - pad)
            ymax = min(h, ymax + pad)
            xmin = max(0, xmin - pad)
            xmax = min(w, xmax + pad)
            img.crop((xmin, ymin, xmax, ymax)).save(shape_path)
            shutil.copy(shape_path, os.path.join("/path/to/output/paper", shape_filename))
            print("Successfully trimmed and copied shape image.")
    except Exception as e:
        print(f"Error post-processing shape image: {e}")

    # ---------------------------------------------------------
    # Save Separate Figure 2: Motion Trajectories
    # ---------------------------------------------------------
    fig_motion = plt.figure(figsize=(4.5, 4.5))
    ax2_ind = fig_motion.add_subplot(111, projection='3d')
    ax2_ind.view_init(elev=_use_elev, azim=_use_azim)
    ax2_ind.dist = 6.5  # Lower = camera closer = sphere fills more of canvas (default is 10)

    # Expand the 3D axes to fill the figure and shift down to reduce bottom whitespace
    ax2_ind.set_position([0.0, -0.05, 1.0, 0.95])

    # Faint wireframe sphere
    ax2_ind.plot_wireframe(x, y, z, alpha=0.15)

    # Custom non-red premium color palette for sample subjects (steel blue, forest green, purple, slate, teal)
    colors = ['#2b5c8f', '#2a7c4e', '#6f4a8a', '#8b7a5d', '#4a7c8a']
    for i, subj in enumerate(sample_subjects):
        mask = (np.array(subject_labels) == subj)
        pts = motion_3d[mask]
        ts = np.array(time_labels)[mask]

        # Slerp-interpolate adjacent points for a smooth geodesic line, including closing the loop
        interp_pts = []
        for idx in range(len(pts) - 1):
            interp_pts.append(slerp(pts[idx], pts[idx+1], num_interp=10)[:-1])
        interp_pts.append(slerp(pts[-1], pts[0], num_interp=10))
        pts_slerped = np.concatenate(interp_pts, axis=0)

        # Plot individual subject trajectories as dashed lines and add a single legend entry
        label = 'Individual (Samples)' if i == 0 else None
        ax2_ind.plot(pts_slerped[:, 0], pts_slerped[:, 1], pts_slerped[:, 2],
                     color=colors[i % len(colors)], alpha=0.45, linewidth=2.5, linestyle='--', label=label)
        sc_ind = ax2_ind.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=ts, cmap='cool',
                                 vmin=0, vmax=1, s=8, alpha=0.70, edgecolor='none', zorder=4)
        ax2_ind.set_facecolor('white')

    # Average motion trajectory (intrinsic mean over all subjects), slerp-smoothed
    ax2_ind.plot(avg_pts_slerped[:, 0], avg_pts_slerped[:, 1], avg_pts_slerped[:, 2],
                 color='red', linewidth=4, zorder=6, label='Average Motion')
    ax2_ind.scatter(avg_motion_3d[0, 0], avg_motion_3d[0, 1], avg_motion_3d[0, 2],
                    color='red', marker='s', s=100, edgecolors='black', zorder=7, label='Start (t=0)')

    ax2_ind.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, framealpha=0.8)

    ax2_ind.set_xlim([-1.05, 1.05])
    ax2_ind.set_ylim([-1.05, 1.05])
    ax2_ind.set_zlim([-1.05, 1.05])
    ax2_ind.axis('off')

    motion_filename = f"motion_trajectories_{parent_name}_{run_name}.png"
    motion_path = os.path.join("/path/to/ACDC_dataset/derivatives/nimosef_v2/alignment_comparison", motion_filename)
    fig_motion.savefig(motion_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig_motion)
    print(f"Saved separate motion trajectories to: {motion_path}")

    # Trim motion trajectories image whitespace and copy to paper
    try:
        img = Image.open(motion_path)
        arr = np.array(img)
        mask = np.any(arr < 250, axis=-1)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if np.any(rows) and np.any(cols):
            ymin, ymax = np.where(rows)[0][0], np.where(rows)[0][-1]
            xmin, xmax = np.where(cols)[0][0], np.where(cols)[0][-1]
            pad = 10
            h, w, _ = arr.shape
            ymin = max(0, ymin - pad)
            ymax = min(h, ymax + pad)
            xmin = max(0, xmin - pad)
            xmax = min(w, xmax + pad)
            img.crop((xmin, ymin, xmax, ymax)).save(motion_path)
            shutil.copy(motion_path, os.path.join("/path/to/output/paper", motion_filename))
            print("Successfully trimmed and copied motion image.")
    except Exception as e:
        print(f"Error post-processing motion image: {e}")


    print("Shape PCA explained variance:", pca.explained_variance_ratio_, 
      "sum:", pca.explained_variance_ratio_.sum())
    print("Motion PCA explained variance:", pca_sphere.explained_variance_ratio_, 
      "sum:", pca_sphere.explained_variance_ratio_.sum())
      
if __name__ == "__main__":
    main()
