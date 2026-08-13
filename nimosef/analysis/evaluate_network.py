import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import inspect

def plot_images(model, dataset, idx, save_folder):
    """
    Evaluates the model on a dataset sample and saves prediction plots.
    Supports both mock and actual models and datasets.
    """
    sample = dataset[idx]
    # Support 6 or 7-value dataset unpack for backward/forward compatibility
    if len(sample) == 7:
        rwc, t0_data, tgt_data, t, transform_info, sample_idx, slice_indices = sample
    else:
        rwc, t0_data, tgt_data, t, transform_info, sample_idx = sample
        slice_indices = torch.zeros(rwc.shape[0], dtype=torch.long)
        
    device = next(model.parameters()).device
    coords = rwc.to(device)
    t_tensor = torch.tensor(t).to(device)
    sample_idx_device = sample_idx.to(device) if isinstance(sample_idx, torch.Tensor) else torch.tensor(sample_idx).to(device)
    slice_indices_device = slice_indices.to(device)
    
    # Call model based on signature
    sig = inspect.signature(model.forward)
    if len(sig.parameters) >= 4:
        out = model(coords, t_tensor, sample_idx_device, slice_indices_device)
    else:
        out = model(coords, t_tensor, sample_idx_device)
        
    seg_pred, intensity_pred = out[0], out[1]
    
    # Save a visualization plot
    os.makedirs(save_folder, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title("Reconstructed Intensity")
    axes[1].set_title("Predicted Segmentation")
    
    plt.savefig(os.path.join(save_folder, "pred_image.png"))
    plt.close()
