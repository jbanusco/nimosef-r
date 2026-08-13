import torch
import torch.nn.functional as F
import dgl
from pytorch3d.ops import knn_points


def dice_loss(pred, target, num_classes, smooth=1e-6, weighted=True, ignore_background=True, apply_softmax=True):
    """
    Multi-class Dice loss with optional weighting and background ignore.
    Args:
        pred (Tensor): (N, C) soft predictions (probabilities) or logits (apply softmax first).
        target (Tensor): (N,) integer class indices.
    """    
    if apply_softmax:
        pred = F.softmax(pred, dim=1)
    
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()

    # Per class computation
    intersection = (pred * target_one_hot).sum(dim=0)
    union = pred.sum(dim=0) + target_one_hot.sum(dim=0)
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

    dice_loss_per_class = 1 - dice_per_class

    if weighted:
        class_voxels = target_one_hot.sum(dim=0)
        if ignore_background:
            class_voxels[0] = 0
        total_voxels = class_voxels.sum()
        class_weights = class_voxels / (total_voxels + smooth)
        class_weights /= torch.clamp(class_weights.sum(), min=smooth)
        return class_weights * dice_loss_per_class
    else:
        if ignore_background:
            dice_loss_per_class[0] = 0
        return dice_loss_per_class


def connectivity_proxy_loss(g):
    deg = g.in_degrees().float()
    return (deg == 0).float().mean()  # fraction of isolated nodes



def compute_graph_connectivity_loss(
    sample_id, boundary_coords, k=5, max_knn_distance=5.0, target_lambda=0.1
):
    """
    Build a KNN graph on boundary coordinates and compute a connectivity
    penalty (fraction of isolated nodes).
    """
    coords_extended = torch.cat((boundary_coords, sample_id.unsqueeze(-1)), dim=1)
    coords_batch = coords_extended.unsqueeze(0).float()

    knn = knn_points(coords_batch, coords_batch, K=k + 1)
    knn_indices = knn.idx[0, :, 1:]  # Remove self-loops (N, k)
    dists = knn.dists[0, :, 1:]  # Squared distances (N, k)

    src = torch.arange(boundary_coords.shape[0], device=boundary_coords.device)
    src = src.unsqueeze(1).repeat(1, k).flatten()  # (N*k,)
    dst = knn_indices.flatten() # (N*k,)

    if max_knn_distance is not None:
        mask_flat = (dists <= max_knn_distance).flatten()
        src, dst = src[mask_flat], dst[mask_flat]
        del mask_flat

    if len(src) == 0:
        print("No valid edges after distance thresholding.")
        return torch.tensor(1.0, device=boundary_coords.device), None

    # Construct a DGL graph
    g = dgl.graph((src, dst), num_nodes=boundary_coords.shape[0])
    
    # EXPLICITLY FREE VRAM
    del knn, coords_batch, coords_extended, knn_indices, dists, src, dst
        
    loss_conn = connectivity_proxy_loss(g)

    return loss_conn, g


def approx_laplacian_on_graph(quantity, coords, g, eps=1e-6, edge_mask=None, normalize_by_length=False):
    """
    Smoothness penalty on a graph (finite-difference laplacian).
    """
    src, dst = g.edges()
    if edge_mask is not None:
        src, dst = src[edge_mask], dst[edge_mask]
    
    if quantity.dim() == 1:
        quantity = quantity.unsqueeze(-1)  # (N, 1)
    
    delta_u = quantity[dst] - quantity[src]  # (E, C)
    if normalize_by_length:
        delta_x = coords[dst] - coords[src]  # (E, 3)
        norm_dx_sq = (delta_x ** 2).sum(dim=1, keepdim=True)  # (E, 1)

        # Sum squared differences across channels normalized by edge length
        per_edge = (delta_u ** 2).sum(dim=1, keepdim=True) / (norm_dx_sq + eps)
    else:
        per_edge = (delta_u ** 2).sum(dim=1, keepdim=True)

    return per_edge.mean()
    

def approx_jacobian_loss(displacement, coords, g, eps=1e-6, penalize_volume=False, max_edges_for_det=50000):
    """
    Folding penalty via edgewise Jacobian determinants.
    """
    src, dst = g.edges()
    # if src.ndim == 1:
    #     src = src.unsqueeze(-1)
    # if dst.ndim == 1:
    #     dst = dst.unsqueeze(-1)

    delta_x = coords[dst] - coords[src]  # (E, 3)
    delta_u = displacement[dst] - displacement[src]  # (E, 3)

    # Get the norm of the differences
    norm_dx = delta_x.norm(dim=1, keepdim=True)

    # Unit direction of delta_x (E, 3)
    delta_x = delta_x / (norm_dx + eps)

    # Approximate derivative magnitude with finite differences (E, 3)
    du = delta_u / (norm_dx + eps)
    
    # Exact determinant calculation via Matrix Determinant Lemma: det(I + u v^T) = 1 + u^T v
    # where u = du and v = delta_x.
    # Ref: https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    detJ = 1.0 + (du * delta_x).sum(dim=-1)

    # Folding penalty (only negatives matter)
    loss_fold = F.relu(-detJ[detJ < 0]).mean() if (detJ < 0).any() else detJ.sum() * 0.0  # keep graph

    # Optional: penalize volume changes
    if penalize_volume:
        loss_vol = ((detJ - 1.0) ** 2).mean()
    else:
        loss_vol = detJ.sum() * 0.0  # zero, keep graph

    return loss_fold, loss_vol
