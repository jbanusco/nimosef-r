import torch
from pytorch3d.ops import knn_points


def compute_distance_pts(pts1, pts2, percentile=0.95, chamfer=False):
    """
    Compute the 95th percentile Hausdorff distance (HF95) between two point clouds.
    
    Args:
        pts1: Tensor of shape (B, N, 3)
        pts2: Tensor of shape (B, M, 3)
    
    Returns:
        hf95: Tensor of shape (B,) containing the HF95 per batch element.
    """
    # Compute nearest neighbors from pts1 to pts2.
    knn1 = knn_points(pts1, pts2, K=1, return_nn=False)
    # Compute nearest neighbors from pts2 to pts1.
    knn2 = knn_points(pts2, pts1, K=1, return_nn=False)
    
    # knn1.dists has shape (B, N, 1) and contains squared distances.
    # Take square root to get Euclidean distances.
    dists1 = knn1.dists.squeeze(-1).sqrt()  # shape: (B, N)
    dists2 = knn2.dists.squeeze(-1).sqrt()  # shape: (B, M)
    
    # Compute the 95th percentile for each batch element.
    hf951 = torch.quantile(dists1, percentile, dim=1)
    hf952 = torch.quantile(dists2, percentile, dim=1)
    
    # The robust Hausdorff distance (HF95) is the maximum of the two directions.
    hf95 = torch.max(hf951, hf952)

    if chamfer:
        # Chamfer distance is the sum of the average distance from pts1->pts2 and pts2->pts1.
        return dists1.mean(dim=1) + dists2.mean(dim=1)        

    return hf95
