import torch
from pytorch3d.ops import knn_points


def compute_pairwise_knn(points, k=10):
    points = points.unsqueeze(0) # (N, D)
    knn = knn_points(points, points, K=k, return_nn=False)  # (B, N, D)
    return knn.dists.squeeze(0), knn.idx.squeeze(0) # (N, K)


def extract_boundary_points(segmentation, coords, sample_id, k=5, max_boundary_points=20000):
    boundary_indices_list = []
    boundary_coords_list = []
    mean_distances = []
    
    # Move unique subject IDs to CPU list once to prevent CPU-GPU sync overhead during loop iteration
    unique_samples = torch.unique(sample_id).cpu().tolist()
    # Budget per subject to keep total bounded
    per_subject_cap = max_boundary_points // max(len(unique_samples), 1)
    
    for subj in unique_samples:
        # Isolate points for this specific subject
        subj_mask = (sample_id == subj)
        subj_indices = subj_mask.nonzero(as_tuple=True)[0]
        
        if len(subj_indices) <= k:
            continue
        
        # Subsample BEFORE KNN to cap the O(N²) memory spike
        if len(subj_indices) > per_subject_cap:
            perm = torch.randperm(len(subj_indices), device=subj_indices.device)[:per_subject_cap]
            subj_indices = subj_indices[perm]
            
        subj_coords = coords[subj_indices]
        subj_seg = segmentation[subj_indices]
        
        # Compute KNN *only* on this subject's (possibly subsampled) points
        coords_batch = subj_coords.unsqueeze(0).float()
        knn = knn_points(coords_batch, coords_batch, K=k + 1)
        knn_indices = knn.idx[0, :, 1:]
        
        # Check boundary condition - simplified using direct advanced indexing
        neighbor_seg = subj_seg[knn_indices]
        is_boundary = (subj_seg.unsqueeze(1) != neighbor_seg).any(dim=1)
        
        # Map boundary indices back to the full batch tensor
        b_idx_in_subj = is_boundary.nonzero(as_tuple=True)[0]
        if len(b_idx_in_subj) > 0:
            actual_b_idx = subj_indices[b_idx_in_subj]
            boundary_indices_list.append(actual_b_idx)
            boundary_coords_list.append(subj_coords[b_idx_in_subj])
            # Optimized distance extraction directly from precomputed KNN distances
            mean_distances.append(torch.sqrt(knn.dists[0, :, 1:].clamp(min=1e-12)).mean())

        # EXPLICITLY FREE VRAM
        del knn, knn_indices, coords_batch, neighbor_seg, is_boundary
        del subj_coords, subj_seg, subj_mask, subj_indices

    if len(boundary_indices_list) == 0:
        # Fallback if no boundaries found
        return torch.tensor([], dtype=torch.long, device=coords.device), torch.empty((0, coords.shape[1]), device=coords.device), torch.tensor(0.0, device=coords.device)

    boundary_indices = torch.cat(boundary_indices_list)
    boundary_coords = torch.cat(boundary_coords_list)
    mean_distance = torch.stack(mean_distances).mean()

    return boundary_indices.long(), boundary_coords, mean_distance
