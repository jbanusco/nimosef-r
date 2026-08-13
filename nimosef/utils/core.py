import torch
import numpy as np
import argparse


def str2bool(v):
    """Parse boolean arguments from CLI."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", "true"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0", "false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def freeze_model_parameters(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model_parameters(model):
    """Unfreeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True




def save_checkpoint(save_path, epoch, model, optimizer, scheduler, loss_tracker):
    """Save training state to a checkpoint file."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss_tracker": loss_tracker,
    }
    
    if isinstance(optimizer, dict):
        checkpoint["optimizer_state_dict"] = {k: v.state_dict() for k, v in optimizer.items()}
    else:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
    if isinstance(scheduler, dict):
        checkpoint["scheduler_state_dict"] = {k: v.state_dict() for k, v in scheduler.items()}
    else:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at epoch {epoch} to {save_path}")


def label_to_color(segmentation, colormap=None):
    """
    Convert a 2D label segmentation (H x W) to a 3-channel RGB image.
    `segmentation` is a numpy array with integer labels.
    `colormap` is a dictionary mapping label -> [R, G, B] (values in 0-255).
    """
    # Example colormap for 4 labels:
    colormap = {
        0: [0, 0, 0],       # background: black
        1: [255, 0, 0],     # label 1: red
        2: [0, 255, 0],     # label 2: green
        3: [0, 0, 255]      # label 3: blue
    }

    H, W = segmentation.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in colormap.items():
        mask = segmentation == label
        color_img[mask] = color
    return color_img


class LossTracker:
    """Track and store losses across epochs."""

    def __init__(self):
        self.losses = []
        self.epoch = []

    def add_loss(self, loss_dict, epoch):
        self.losses.append(loss_dict)
        self.epoch.append(epoch)

    def save_losses(self, filename):
        torch.save({"epoch": self.epoch, "losses": self.losses}, filename)

    def load_losses(self, filename):
        checkpoint = torch.load(filename, weights_only=False)
        self.epoch = checkpoint["epoch"]
        self.losses = checkpoint["losses"]
