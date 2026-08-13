import os
from nimosef.training.train import train_model
from nimosef.training.args import get_train_parser
import torch

torch.autograd.set_detect_anomaly(False)

def main():
    data_path = os.environ.get("DATA_PATH", "/path/to/ACDC_dataset")
    os.environ["DATA_PATH"] = data_path
    manifest_file = os.path.join(data_path, "derivatives", "manifests_nimosef", "dataset_manifest.json")
    save_folder = os.path.join(data_path, "derivatives", "nimosef_v2", "manifold_overfit_muon")

    # Load parser with defaults
    parser = get_train_parser()
    args = parser.parse_args([
        "--data_folder", data_path,
        "--split_file", manifest_file,
        "--save_folder", save_folder,
        "--resume_from_checkpoint", "True",
        "--load_model", "True",
        "--batch_size", "3",
        "--sample_percentage", "0.3",
        "--preload_data", "False",
        "--prefetch_factor", "1",
        "--num_workers", "1",
        "--num_epochs", "1000",
        "--epochs_to_evaluate_validation", "10000",
        "--validation_epochs", "20",
        "--lambda_rec", "10.0",
        "--lambda_seg", "2.0",
        "--lambda_reg", "0.0",
        "--lambda_dsp", "2.0",
        "--lambda_reg_dsp", "1.0",
        "--lambda_jacobian", "5.0",
        "--lambda_vol", "0.01",
        "--lambda_graph_conn", "0.0",
        "--lambda_smoothness", "0.1",
        "--warmup_epochs", "5",
        "--lr", "0.0001",
        "--lr_shape_code", "0.005",
        "--lr_scheduler_step", "10",
        "--lr_scheduler_gamma", "0.95",
        "--skip_correction", "False",
        "--use_trig_encoding", "True"
    ])
    print(">>> Starting training on", args.data_folder)
    train_model(args)

    # Save symlink to last model for inference
    import glob, shutil
    run_folders = sorted(glob.glob(os.path.join(save_folder, "experiment_*")))
    if run_folders:
        final_model = os.path.join(run_folders[-1], "model.pth")
        link_path = os.path.join(save_folder, "latest_model.pth")
        if os.path.exists(final_model):
            shutil.copyfile(final_model, link_path)
            print(f">>> Copied latest model to {link_path}")

if __name__ == "__main__":
    main()
