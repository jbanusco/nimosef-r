import os
import sys
import torch
import geoopt
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nimosef.models.nimosef import MultiHeadNetwork
from nimosef.data.dataset import NiftiDataset
from nimosef.training.generate_results import generate_reconstructed_images

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

def main():
    data_path = os.environ.get("DATA_PATH", "/path/to/ACDC_dataset")
    os.environ["DATA_PATH"] = data_path
    manifest_file = os.path.join(data_path, "derivatives", "manifests_nimosef", "dataset_manifest.json")

    # NIMOSEF-R checkpoint (the only one this script makes sense for -- it
    # computes a Poincare-ball Frechet mean, a Riemannian-only concept).
    latest_run = os.path.join(data_path, "derivatives", "nimosef_v2", "muon_riemann_model",
                               "experiment_20260701_142855")
    model_path = os.path.join(latest_run, "checkpoint_last.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(latest_run, "model.pth")
        
    print(f"Loading model from {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
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
            "use_trig_encoding": False
        }

    # Extract sizes from state_dict to confirm
    num_subjects = state_dict["shape_code.weight"].shape[0]
    
    # Determine subject slices
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
        use_trig_encoding=hparams.get("use_trig_encoding", False)
    ).to(device)
    
    model.load_state_dict(state_dict, strict=False)
    
    print("Calculating Average Shape (True Fréchet Mean via Riemannian Optimization)...")
    with torch.no_grad():
        shape_codes = model.shape_code.weight.detach()
        manifold = geoopt.PoincareBall(c=1.0)
    
    avg_shape_code = compute_frechet_mean(shape_codes, manifold)
    
    with torch.no_grad():
        # Overwrite the shape code for subject 0 with the average shape
        model.shape_code.weight[0].copy_(avg_shape_code)
        
    # We must skip correction for the "average" subject to avoid using subject 0's specific misalignment
    model.skip_correction = True
    model.eval()
    
    print("Loading dataset for geometry reference...")
    dataset = NiftiDataset(manifest_file, mode="val", preload_data=False)
    
    # Create an output directory for the average heart
    output_dir = os.path.join(latest_run, "average_heart_reconstruction")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating synthetic Average Heart over a full cardiac cycle...")
    print(f"Results will be saved to: {output_dir}")
    
    # Use generate_reconstructed_images in synthetic mode for Subject 0
    # is_synthetic=True forces it to use a generic clean 128x128x10 grid and 50 time steps
    generate_reconstructed_images(
        model=model,
        val_dataset=dataset,
        save_folder=output_dir,
        res_factor_z=1.0,
        reprocess=True,
        num_subjects=1,          # Only generate subject 0 (which is now the average)
        is_synthetic=True,
        save_motion_corrected=False
    )
    
    # Rename output for clarity
    old_patient_id = dataset.patients[0]
    out_subj_dir = os.path.join(output_dir, old_patient_id)
    new_subj_dir = os.path.join(output_dir, "Average_Heart")
    
    import shutil
    if os.path.exists(out_subj_dir):
        if os.path.exists(new_subj_dir):
            shutil.rmtree(new_subj_dir)
        os.rename(out_subj_dir, new_subj_dir)
        print(f"\nDone! Your Average Heart 4D NIfTI files are located at:\n{new_subj_dir}")
    else:
        print("\nGeneration finished, but could not rename the folder.")

if __name__ == "__main__":
    main()
