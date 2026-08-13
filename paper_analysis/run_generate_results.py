import os
import sys

# Ensure local nimosef module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nimosef.training.args import get_inference_parser
from nimosef.training.generate_results import main as generate_results_main

def main():
    # Provide the dataset and model paths for testing/debugging
    data_path = os.environ.get("DATA_PATH", "/path/to/ACDC_dataset")
    os.environ["DATA_PATH"] = data_path
    
    manifest_file = os.path.join(data_path, "derivatives", "manifests_nimosef", "dataset_manifest.json")
    target_mode = "train" # Options: 'train', 'val', 'test'

    # NIMOSEF-E (Euclidean latents). For NIMOSEF-R instead, point experiment_folder
    # at "muon_riemann_model"/"experiment_20260701_142855" and add "--riemannian", "True"
    # to the sys.argv list below (default is False, i.e. NIMOSEF-E).
    experiment_folder = os.path.join(data_path, "derivatives", "nimosef_v2", "baseline_muon")
    checkpoint_path = os.path.join(experiment_folder, "experiment_20260625_152351", "checkpoint_last.pth")

    save_folder = experiment_folder
    save_rec_folders = os.path.join(experiment_folder, "reconstructions")

    # We fake sys.argv so get_inference_parser() uses these values
    sys.argv = [
        "generate_results.py",
        "--data_folder", data_path,
        "--split_file", manifest_file,
        "--save_folder", save_folder,
        "--save_rec_folders", save_rec_folders,
        "--model_to_rec", checkpoint_path,
        "--mode", target_mode,               # Options: 'train', 'val', 'test'. Change it to select different subjects
        "--res_factor_z", "1",
        "--save_motion_corrected", "True",
        "--overwrite_imgs", "True",     # Set to True so it recalculates even if files exist
        
        # Model architecture parameters (make sure they match the trained model)
        "--num_labels", "4",
        "--latent_size", "128",
        "--motion_size", "64",
        "--hidden_size", "128",
        "--num_res_layers", "8",
        "--linear_head", "True",
        "--skip_correction", "False"     # Set to False to enable slice correction prediction
    ]

    print(f">>> Starting generate_results for {data_path}")
    print(f">>> Mode: {sys.argv[sys.argv.index('--mode')+1]}")
    print(f">>> Checkpoint: {checkpoint_path}")

    # Call main directly since it parses args from sys.argv
    generate_results_main()

if __name__ == "__main__":
    main()
