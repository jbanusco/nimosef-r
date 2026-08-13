import argparse
from nimosef.utils.core import str2bool  # you already had this helper


def get_train_parser():
    parser = argparse.ArgumentParser(description="Train the NIMOSEF model")

    # Dataset
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--split_file", type=str, required=True, help="Path to JSON split file")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_percentage", type=float, default=1.0)
    parser.add_argument("--preload_data", type=str2bool, default=True)
    parser.add_argument("--debug", action="store_true", help="Use a subset of ~10 subjects for debugging")
    parser.add_argument("--num_labels", type=int, default=4)

    # Model
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--motion_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_res_layers", type=int, default=8)
    parser.add_argument("--linear_head", type=str2bool, default=True)
    parser.add_argument("--skip_correction", type=str2bool, default=True)
    parser.add_argument("--use_trig_encoding", type=str2bool, default=False)
    parser.add_argument("--riemannian", type=str2bool, default=False, help="If True, constrain shape codes to the Poincare ball and motion codes to the hypersphere (NIMOSEF-R). If False, use unconstrained Euclidean latents (NIMOSEF-E).")

    # Training
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--epochs_to_evaluate_validation", type=int, default=100)
    parser.add_argument("--validation_epochs", type=int, default=50)    

    # Loss weights (basic)
    parser.add_argument("--lambda_rec", type=float, default=2.0)
    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--lambda_dsp", type=float, default=0.5)
    parser.add_argument("--lambda_reg_dsp", type=float, default=1.0)
    parser.add_argument("--lambda_jacobian", type=float, default=1.0)
    parser.add_argument("--lambda_vol", type=float, default=0.01)
    parser.add_argument("--lambda_graph_conn", type=float, default=0.0)
    parser.add_argument("--lambda_smoothness", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--max_dsp_weight", type=float, default=1.0)

    # Optimizer & scheduler
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_shape_code", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler_step", type=int, default=10)
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.95)

    # Saving
    parser.add_argument("--save_folder", type=str, required=True, help="Base folder where to save outputs")
    parser.add_argument("--load_model", type=str2bool, default=False)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--checkpoint_filename", type=str, default=None)

    return parser


def get_inference_parser():
    parser = argparse.ArgumentParser(description="Inference training for NIMOSEF")

    # Dataset
    parser.add_argument("--data_folder", type=str, required=False, help="Path to the dataset folder")
    parser.add_argument("--split_file", type=str, required=False, help="Path to JSON split file")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--preload_data", type=str2bool, default=True)
    parser.add_argument("--num_labels", type=int, default=4)

    # Model
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--motion_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_res_layers", type=int, default=8)
    parser.add_argument("--linear_head", type=str2bool, default=True)
    parser.add_argument("--skip_correction", type=str2bool, default=True)
    parser.add_argument("--use_trig_encoding", type=str2bool, default=False)
    parser.add_argument("--riemannian", type=str2bool, default=False, help="If True, constrain shape codes to the Poincare ball and motion codes to the hypersphere (NIMOSEF-R). If False, use unconstrained Euclidean latents (NIMOSEF-E).")

    # Inference training
    parser.add_argument("--validation_epochs", type=int, default=100)
    parser.add_argument("--initial_model_path", type=str, required=False, help="Path to pretrained model")
    parser.add_argument("--validation_model_path", type=str, default=None)
    parser.add_argument("--lr_shape_code", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler_step", type=int, default=10)
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.95)

    # Loss weights (basic)
    parser.add_argument("--lambda_rec", type=float, default=2.0)
    parser.add_argument("--lambda_seg", type=float, default=0.0)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--lambda_dsp", type=float, default=0.5)
    parser.add_argument("--lambda_reg_dsp", type=float, default=1.0)
    parser.add_argument("--lambda_jacobian", type=float, default=1.0)
    parser.add_argument("--lambda_vol", type=float, default=0.01)
    parser.add_argument("--lambda_graph_conn", type=float, default=0.0)
    parser.add_argument("--lambda_smoothness", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--max_dsp_weight", type=float, default=1.0)

    # Saving
    parser.add_argument("--save_folder", type=str, required=False, help="Base folder where to save outputs")
    parser.add_argument("--res_factor_z", type=float, default=1.0, help="Resolution factor for z axis")
    parser.add_argument("--save_rec_folders", type=str, default=None, help="Folder to save reconstructions")
    parser.add_argument("--overwrite_imgs", type=str2bool, default=False)
    parser.add_argument("--save_motion_corrected", type=str2bool, default=False)
    parser.add_argument("--model_to_rec", type=str, default=None)

    return parser
