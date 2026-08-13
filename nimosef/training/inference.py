import os
import time
import torch
import numpy as np
import datetime
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import geoopt

from nimosef.models.nimosef import MultiHeadNetwork, SliceCorrection
from nimosef.data.dataset import NiftiDataset, nifti_collate_fn
from nimosef.training.logger_utils import TrainingLogger, LossTracker
from nimosef.training.schedulers import ClampedStepLR
from nimosef.training.args import get_inference_parser
from nimosef.losses.base import dice_loss
from nimosef.losses.composite import CompositeLoss
from nimosef.utils.core import save_checkpoint, label_to_color


def inference_training(args, model=None, split=None, device=None):
    """
    Inference training loop: freeze network, optimize shape codes.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = 4

    hparams = {
        "latent_size": args.latent_size,
        "motion_size": args.motion_size,
        "hidden_size": args.hidden_size,
        "num_res_layers": args.num_res_layers,
        "linear_head": args.linear_head,
        "use_trig_encoding": args.use_trig_encoding,
        "lr": args.lr_shape_code,
        "lr_shape_code": args.lr_shape_code,
        "lambda_rec": args.lambda_rec,
        "lambda_seg": args.lambda_seg,
        "lambda_reg": args.lambda_reg,
        "lambda_dsp": args.lambda_dsp,
        "lambda_reg_dsp": args.lambda_reg_dsp,
        "lambda_jacobian": args.lambda_jacobian,
        "lambda_vol": args.lambda_vol,
        "lambda_graph_conn": args.lambda_graph_conn,
        "lambda_smoothness": args.lambda_smoothness,
        "warmup_epochs": args.warmup_epochs,
        "max_dsp_weight": args.max_dsp_weight,
    }

    # Ensure we use the correct split (passed via argument or from args)
    mode = split if split is not None else args.mode
    dataset = NiftiDataset(args.split_file, mode=mode, preload_data=args.preload_data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,        
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        collate_fn=nifti_collate_fn,
    )
    # Setup experiment folders and file names (you may re-use your base save folder)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"inference_{run_id}"    
    base_save_folder = args.save_folder    
    save_folder = os.path.join(base_save_folder, run_name)
    os.makedirs(save_folder, exist_ok=True)
    print(f"Inference results will be saved to {save_folder}")
    
    # Save patient list for validation
    torch.save({'patient_list': dataset.patients}, os.path.join(save_folder, 'validation_patient_list.pth'))
    num_subjects = len(dataset)
    true_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    
    subject_slices = {}
    for subj_idx, subj in enumerate(true_dataset.subjects):
        subject_slices[subj_idx] = true_dataset.get_subject_num_slices(subj_idx)

    if model is None:
        num_subjects = len(dataset)
        model = MultiHeadNetwork(subject_slices, num_subjects, num_labels,
                                 args.latent_size, args.motion_size,
                                 hidden_size=args.hidden_size,
                                 num_res_layers=args.num_res_layers,
                                 linear_head=args.linear_head,
                                 skip_correction=args.skip_correction,
                                 riemannian=getattr(args, "riemannian", True))
        
        data_state = torch.load(args.initial_model_path, map_location=device)
        if "model_state_dict" in data_state:
            model_state = data_state["model_state_dict"]
        else:
            model_state = data_state
        # Drop shape codes
        if "shape_code.weight" in model_state:
            del model_state["shape_code.weight"]
        # Drop all slice correction embeddings
        slice_keys = [k for k in model_state.keys() if k.startswith("slice_correction.subject_tables.") or k == "slice_correction.table.weight"]
        for k in slice_keys:
            del model_state[k]
        model.load_state_dict(model_state, strict=False)
    
    else:
        # Re-init the shape embeddings and slice embeddings
        # Shape codes - Riemannian aware if possible
        if hasattr(model, 'manifold'):
            from nimosef.models.nimosef import RiemannianEmbedding
            model.shape_code = RiemannianEmbedding(num_subjects, args.latent_size, model.manifold).to(device)
        else:
            model.shape_code = torch.nn.Embedding(num_subjects, args.latent_size).to(device)
            torch.nn.init.normal_(model.shape_code.weight.data, 0.0, 0.01 / math.sqrt(args.latent_size))

        # Slice corrections
        model.slice_correction = SliceCorrection(subject_slices).to(device)

    model.to(device)

    # Freeze everything except shape codes and slice corrections
    for name, param in model.named_parameters():
        if "shape_code" not in name and "slice_correction" not in name:
            param.requires_grad = False

    euclidean_optimizer = torch.optim.AdamW([
        {'params': model.slice_correction.parameters(), 'lr': args.lr_shape_code},
    ], weight_decay=1e-6)
    
    if hasattr(model, 'manifold'):
        riemannian_optimizer = geoopt.optim.RiemannianAdam(
            model.shape_code.parameters(),
            lr=args.lr_shape_code
        )
    else:
        riemannian_optimizer = torch.optim.AdamW(
            model.shape_code.parameters(),
            lr=args.lr_shape_code
        )

    euclidean_scheduler = ClampedStepLR(
        euclidean_optimizer,
        step_size=args.lr_scheduler_step,
        gamma=args.lr_scheduler_gamma,
        min_lrs=[1e-4, 2e-4],
    )
    
    riemannian_scheduler = torch.optim.lr_scheduler.StepLR(
        riemannian_optimizer,
        step_size=args.lr_scheduler_step,
        gamma=args.lr_scheduler_gamma,
    )
    
    opt_dict = {"euclidean": euclidean_optimizer, "riemannian": riemannian_optimizer}
    sched_dict = {"euclidean": euclidean_scheduler, "riemannian": riemannian_scheduler}

    writer = SummaryWriter(log_dir=os.path.join(save_folder, "inference_logs"))
    logger = TrainingLogger(log_filename=os.path.join(save_folder, "inference.log"))
    tracker = LossTracker()

    # === Loss ===
    loss_module = CompositeLoss(is_test=True, device=device, hp_dict=hparams, manifold=model.manifold)

    # === Resume from checkpoint ===
    start_epoch = 0
    ckpt_file = getattr(args, "resume_checkpoint", None)
    if ckpt_file and os.path.exists(ckpt_file):
        checkpoint = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load dual optimizers if present
        if isinstance(checkpoint.get("optimizer_state_dict"), dict) and "euclidean" in checkpoint["optimizer_state_dict"]:
            euclidean_optimizer.load_state_dict(checkpoint["optimizer_state_dict"]["euclidean"])
            riemannian_optimizer.load_state_dict(checkpoint["optimizer_state_dict"]["riemannian"])
            
        tracker = checkpoint.get("loss_tracker", tracker)
        start_epoch = checkpoint["epoch"]
        print(f"Resumed inference training from {ckpt_file} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.validation_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_metrics = {
            'Total_loss': 0.0,
            'L_intensity': 0.0,
            'L_seg': 0.0,
            'L_latent': 0.0,
            'L_disp': 0.0,
            'L_disp_reg': 0.0,
            'L_J': 0.0,
            'L_Jfold': 0.0,
            "L_Jvol": 0.0,
            'L_graph': 0.0,
            'L_smooth_int': 0.0,
            'L_smooth_seg': 0.0,
            'L2_error': 0.0,
            'LV_dice': 0.0,
            'MYO_dice': 0.0,
            'RV_dice': 0.0,
        }
        epoch_batches = 0

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Inference Epoch {epoch}")):
            euclidean_optimizer.zero_grad(set_to_none=True)
            riemannian_optimizer.zero_grad(set_to_none=True)
            
            # Unpack batch data
            coords, t0_data, tgt_data, t, transform_info, sample_id, slice_indices = batch
            t0_intensity, t0_seg, t0_mask = t0_data
            tgt_intensity, tgt_seg, tgt_mask = tgt_data

            # Sanity checks
            assert tgt_seg.max() < args.num_labels, f"Invalid seg label: {tgt_seg.unique()}"
            assert t0_seg.max() < args.num_labels, f"Invalid t0 seg label: {t0_seg.unique()}"
            assert sample_id.max() < model.shape_code.num_embeddings, f"Sample ID out of range: {sample_id.max()} vs {model.shape_code.num_embeddings}"

            # Send to device (t_gpu stays on device, others are chunked to device inside the loop)
            # We keep large tensors on CPU and only move chunks to GPU
            t_gpu = t.to(device)
            sample_id_gpu = sample_id.to(device)
            slice_indices_gpu = slice_indices.to(device)

            num_points = coords.shape[0]
            chunk_size = 65536 # ~65k points per chunk to stay well within 24GB
            
            # Accumulated predictions for visualization/metrics
            all_seg_pred_t = torch.zeros((num_points, num_labels), device=device)
            all_int_pred_t = torch.zeros((num_points, 1), device=device)
            all_seg_pred_t0 = torch.zeros((num_points, num_labels), device=device)
            all_int_pred_t0 = torch.zeros((num_points, 1), device=device)
            
            epoch_loss_val = 0.0
            epoch_components = {}
            
            # Common tensors for this subject
            centers = transform_info['centers'][0].to(device)
            affine = transform_info['affines'][0].to(device)
            indices = torch.cat(transform_info['indices'], dim=0).to(device)
            original_resolution = torch.linalg.norm(affine[:3, :3], axis=0)

            # Process in chunks
            for i in range(0, num_points, chunk_size):
                end_i = min(i + chunk_size, num_points)
                c_coords = coords[i:end_i].to(device)
                
                # Chunked indices and time
                c_sample_id = sample_id_gpu[i:end_i]
                c_slice_indices = slice_indices_gpu[i:end_i]
                c_t = t_gpu[i:end_i].unsqueeze(-1)
                
                # t0 forward
                t0_vec = torch.zeros(c_coords.shape[0], 1, device=device)
                seg_c0, int_c0, disp_c0, _ = model(c_coords, t0_vec, c_sample_id, c_slice_indices)
                
                # t forward
                seg_ct, int_ct, disp_ct, _ = model(c_coords, c_t, c_sample_id, c_slice_indices)
                
                # Collect for metrics/viz (DETACH to prevent graph accumulation leak)
                all_seg_pred_t[i:end_i] = seg_ct.detach()
                all_int_pred_t[i:end_i] = int_ct.detach()
                all_seg_pred_t0[i:end_i] = seg_c0.detach()
                all_int_pred_t0[i:end_i] = int_c0.detach()

                # Chunked targets
                c_t0_seg = t0_seg[i:end_i].to(device)
                c_t0_int = t0_intensity[i:end_i].to(device)
                c_t0_mask = t0_mask[i:end_i].to(device)
                c_tgt_seg = tgt_seg[i:end_i].to(device)
                c_tgt_int = tgt_intensity[i:end_i].to(device)
                c_tgt_mask = tgt_mask[i:end_i].to(device)

                # Loss for this chunk
                h = model.shape_code(torch.unique(sample_id_gpu))
                preds_t0_c = {'seg_pred': seg_c0, 'intensity_pred': int_c0, 'displacement': disp_c0, 'h': h, 'batch_size': args.batch_size}
                preds_t_c = {'seg_pred': seg_ct, 'intensity_pred': int_ct, 'displacement': disp_ct, 'h': h, 'batch_size': args.batch_size}
                targets_t0_c = {'segmentation': c_t0_seg, 'intensity': c_t0_int, 'mask': c_t0_mask, 'batch_size': args.batch_size}
                targets_t_c = {'segmentation': c_tgt_seg, 'intensity': c_tgt_int, 'mask': c_tgt_mask, 'batch_size': args.batch_size}
            # Send to device (t_gpu stays on device, others are chunked to device inside the loop)
            # We keep large tensors on CPU and only move chunks to GPU
            t_gpu = t.to(device)
            sample_id_gpu = sample_id.to(device)
            slice_indices_gpu = slice_indices.to(device)

            num_points = coords.shape[0]
            chunk_size = 65536 # ~65k points per chunk to stay well within 24GB
            
            # Accumulated predictions for visualization/metrics
            all_seg_pred_t = torch.zeros((num_points, num_labels), device=device)
            all_int_pred_t = torch.zeros((num_points, 1), device=device)
            all_seg_pred_t0 = torch.zeros((num_points, num_labels), device=device)
            all_int_pred_t0 = torch.zeros((num_points, 1), device=device)
            
            epoch_loss_val = 0.0
            epoch_components = {}
            
            # Common tensors for this subject
            centers = transform_info['centers'][0].to(device)
            affine = transform_info['affines'][0].to(device)
            indices = torch.cat(transform_info['indices'], dim=0).to(device)
            original_resolution = torch.linalg.norm(affine[:3, :3], axis=0)

            # Process in chunks
            for i in range(0, num_points, chunk_size):
                end_i = min(i + chunk_size, num_points)
                c_coords = coords[i:end_i].to(device)
                
                # Chunked indices and time
                c_sample_id = sample_id_gpu[i:end_i]
                c_slice_indices = slice_indices_gpu[i:end_i]
                c_t = t_gpu[i:end_i].unsqueeze(-1)
                
                # t0 forward
                t0_vec = torch.zeros(c_coords.shape[0], 1, device=device)
                seg_c0, int_c0, disp_c0, _ = model(c_coords, t0_vec, c_sample_id, c_slice_indices)
                
                # t forward
                seg_ct, int_ct, disp_ct, _ = model(c_coords, c_t, c_sample_id, c_slice_indices)
                
                # Collect for metrics/viz (DETACH to prevent graph accumulation leak)
                all_seg_pred_t[i:end_i] = seg_ct.detach()
                all_int_pred_t[i:end_i] = int_ct.detach()
                all_seg_pred_t0[i:end_i] = seg_c0.detach()
                all_int_pred_t0[i:end_i] = int_c0.detach()

                # Chunked targets
                c_t0_seg = t0_seg[i:end_i].to(device)
                c_t0_int = t0_intensity[i:end_i].to(device)
                c_t0_mask = t0_mask[i:end_i].to(device)
                c_tgt_seg = tgt_seg[i:end_i].to(device)
                c_tgt_int = tgt_intensity[i:end_i].to(device)
                c_tgt_mask = tgt_mask[i:end_i].to(device)

                # Loss for this chunk
                h = model.shape_code(torch.unique(sample_id_gpu))
                preds_t0_c = {'seg_pred': seg_c0, 'intensity_pred': int_c0, 'displacement': disp_c0, 'h': h, 'batch_size': args.batch_size}
                preds_t_c = {'seg_pred': seg_ct, 'intensity_pred': int_ct, 'displacement': disp_ct, 'h': h, 'batch_size': args.batch_size}
                targets_t0_c = {'segmentation': c_t0_seg, 'intensity': c_t0_int, 'mask': c_t0_mask, 'batch_size': args.batch_size}
                targets_t_c = {'segmentation': c_tgt_seg, 'intensity': c_tgt_int, 'mask': c_tgt_mask, 'batch_size': args.batch_size}

                loss_chunk, components_chunk = loss_module(c_sample_id, c_coords, original_resolution,
                                                           preds_t0_c, preds_t_c, targets_t0_c, targets_t_c, epoch=epoch)
                
                # Scale loss by chunk weight for accumulation
                weight = (end_i - i) / num_points
                (loss_chunk * weight).backward()

                epoch_loss_val += loss_chunk.item() * weight
                for k, v in components_chunk.items():
                    epoch_components[k] = epoch_components.get(k, 0.0) + v * weight

            # Step optimizers after processing all chunks for this subject
            euclidean_optimizer.step()
            riemannian_optimizer.step()

            with torch.no_grad():
                # Correct dice/l2 error using full subject preds
                # Need target segs on GPU for dice
                tgt_seg_gpu = tgt_seg.to(device)
                tgt_int_gpu = tgt_intensity.to(device)
                
                lv_dice, myo_dice, rv_dice = 1 - dice_loss(all_seg_pred_t, tgt_seg_gpu, num_labels, ignore_background=True, weighted=False, apply_softmax=True)[1:]
                l2_error = ((all_int_pred_t.squeeze() - tgt_int_gpu.squeeze())**2).mean()

                # Update dictionary accumulators:
                epoch_metrics['Total_loss'] += epoch_loss_val
                for key, val in epoch_components.items():
                    epoch_metrics[key] += val.item() if isinstance(val, torch.Tensor) else val

                epoch_metrics["LV_dice"] += lv_dice.item()
                epoch_metrics["MYO_dice"] += myo_dice.item()
                epoch_metrics["RV_dice"] += rv_dice.item()
                epoch_metrics['L2_error'] += l2_error.item()

            epoch_batches += 1

        # === End of epoch ===
        for k in epoch_metrics:
            epoch_metrics[k] /= epoch_batches

        tracker.add_loss(epoch_metrics, epoch)
        writer.add_scalar("Inference_Loss/Total_loss", epoch_metrics["Total_loss"], epoch)
        writer.add_scalar("Inference_Dice/LV", epoch_metrics["LV_dice"], epoch)
        writer.add_scalar("Inference_Dice/MYO", epoch_metrics["MYO_dice"], epoch)
        writer.add_scalar("Inference_Dice/RV", epoch_metrics["RV_dice"], epoch)

        # === Save checkpoint ===
        if (epoch + 1) % getattr(args, "checkpoint_epochs", 20) == 0:
            ckpt_path = os.path.join(save_folder, f"inference_checkpoint_last.pth")
            save_checkpoint(ckpt_path, epoch+1, model, opt_dict, sched_dict, tracker)
        
        # === Print/logging ===
        if (epoch % getattr(args, "print_epochs", 5)) == 0:
            # Log the averages to TensorBoard:
            for key, avg_val in epoch_metrics.items():
                writer.add_scalar(f'Inference_Loss/{key}', avg_val, epoch)

            str_to_print = [f"{key}: {val:.4f}" for key, val in epoch_metrics.items()]
            print(f"Epoch {epoch}: " + ", ".join(str_to_print))

            # - Learning rates
            for i, param_group in enumerate(euclidean_optimizer.param_groups):
                writer.add_scalar(f'Inference_LR/euclidean_group_{i}', param_group['lr'], epoch)
            for i, param_group in enumerate(riemannian_optimizer.param_groups):
                writer.add_scalar(f'Inference_LR/riemannian_group_{i}', param_group['lr'], epoch)
            
            # - Histograms of weights and gradients
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(name, param, epoch)
                    try:
                        if param.grad is not None:
                            writer.add_histogram(f'Inference_Grad/{name}_grad', param.grad, epoch)
                        else:
                            print(f"No gradient for parameter: {name}")
                    except Exception as e:
                        print(f"Error logging gradient for parameter: {name}")
                        print(e)

            # - Predicted images and visualize mid-
            with torch.no_grad():
                # pick one subject ID from the batch
                subject_id = sample_id[0].item()
                idx_subject = (sample_id == subject_id).nonzero(as_tuple=True)[0]

                # slice tensors for this subject only (no idx_points!)
                subj_indices = indices # In inference batch size 1, indices is already for this subject
                subj_int_pred = all_int_pred_t0
                subj_seg_pred = torch.argmax(all_seg_pred_t0, dim=1)
                subj_t0_intensity = t0_intensity
                subj_t0_seg = t0_seg

                # reconstruct shape from max index
                x, y, z = subj_indices.max(0).values.int().tolist()
                x, y, z = x + 1, y + 1, z + 1
                visualize_mid = z // 2

                # allocate full arrays
                pred_im_final = np.zeros((x, y, z))
                pred_seg_final = np.zeros((x, y, z))
                t0_im = np.zeros((x, y, z))
                t0_seg_im = np.zeros((x, y, z))

                # fill arrays using voxel indices
                coords_np = subj_indices.cpu().numpy().astype(np.int16)
                pred_im_final[coords_np[:,0], coords_np[:,1], coords_np[:,2]] = subj_int_pred.squeeze().cpu().numpy()
                pred_seg_final[coords_np[:,0], coords_np[:,1], coords_np[:,2]] = subj_seg_pred.squeeze().cpu().numpy()
                t0_im[coords_np[:,0], coords_np[:,1], coords_np[:,2]] = subj_t0_intensity.squeeze().cpu().numpy()
                t0_seg_im[coords_np[:,0], coords_np[:,1], coords_np[:,2]] = subj_t0_seg.squeeze().cpu().numpy()

                pred_seg_color = label_to_color(pred_seg_final[..., visualize_mid])
                t0_seg_color = label_to_color(t0_seg_im[..., visualize_mid])

                # concat plots
                intensity_plot = np.concatenate((pred_im_final[..., visualize_mid],
                                                t0_im[..., visualize_mid]), axis=-1)
                segmentation_plot = np.concatenate((pred_seg_color, t0_seg_color), axis=1)

            # Convert the label image to a color image:
            writer.add_image('Inference/Reconstruction', intensity_plot, epoch, dataformats='HW')
            writer.add_image('Inference/Segmentation', segmentation_plot.transpose(2, 0, 1), epoch, dataformats='CHW')

            # Log epoch processing time and speed
            epoch_time = time.time() - epoch_start_time
            writer.add_scalar('Inference_Epoch/Time_sec', epoch_time, epoch)
            writer.add_scalar('Inference_Epoch/Batches_per_sec', epoch_batches / epoch_time, epoch)
            print(f"Epoch {epoch} took {epoch_time:.2f} seconds, {epoch_batches/epoch_time:.2f} batches/sec")

        euclidean_scheduler.step()
        riemannian_scheduler.step()
        logger.on_epoch_end(epoch, logs=epoch_metrics["Total_loss"])

    # Take as score the average dice of the last epoch
    score = (epoch_metrics['LV_dice'] + epoch_metrics['MYO_dice'] + epoch_metrics['RV_dice']) / 3
    writer.add_hparams(hparams, {'hparam/score': score}, run_name=run_name)
    print(f"Inference finished. Score: {score}")

    # Save the model as validation
    if hasattr(args, 'validation_model_path'):
        validation_model_filename = args.validation_model_path
    else:
        validation_model_filename = None

    if validation_model_filename is None:
        validation_model_filename = os.path.join(save_folder, 'validation_model.pth')
    torch.save(model.state_dict(), validation_model_filename)

    # Save final model
    ckpt_path = os.path.join(save_folder, f"inference_checkpoint_final.pth")
    save_checkpoint(ckpt_path, epoch+1, model, opt_dict, sched_dict, tracker)
    tracker.save_losses(os.path.join(save_folder, "losses_inference.json"))

    writer.flush()
    writer.close()

    print(f"Validation model saved to {validation_model_filename}")

    return model


def main():
    # Load parser with defaults
    parser = get_inference_parser()
    args = parser.parse_args()
    print(">>> Starting inference on", args.data_folder)
    inference_training(args)


if __name__ == '__main__':
    main()