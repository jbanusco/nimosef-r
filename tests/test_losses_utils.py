import torch
from pytorch3d.loss import chamfer_distance

from nimosef.losses.utils import (
    compute_pairwise_knn,
    extract_boundary_points,
)


def test_chamfer_distance_sanity():
    # Two identical point clouds
    pts1 = torch.tensor([[[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]]])
    pts2 = pts1.clone()
    loss, _ = chamfer_distance(pts1, pts2)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    # Shifted version of pts2
    shift = torch.tensor([[[1.0, 0.0, 0.0]]])
    pts2_shifted = pts2 + shift
    loss_shifted, _ = chamfer_distance(pts1, pts2_shifted)
    assert loss_shifted > loss

    # Larger shift → larger Chamfer
    pts2_big_shift = pts2 + torch.tensor([[[5.0, 0.0, 0.0]]])
    loss_big, _ = chamfer_distance(pts1, pts2_big_shift)
    assert loss_big > loss_shifted


def test_compute_pairwise_knn_shapes():
    points = torch.rand(8, 3)
    dists, idx = compute_pairwise_knn(points, k=4)
    assert dists.shape == idx.shape
    assert dists.shape[1] == 4


def test_extract_boundary_points_detects_boundary():
    N = 12
    coords = torch.rand(N, 3)
    seg = torch.cat([torch.zeros(N//2), torch.ones(N//2)]).long()
    sample_id = torch.zeros(N, dtype=torch.long)
    idx, bcoords, mdist = extract_boundary_points(seg, coords, sample_id, k=3)
    assert idx.ndim == 1
    assert bcoords.shape[1] == 3
    assert mdist > 0


def test_compute_pairwise_knn():
    points = torch.tensor([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
    dists, idx = compute_pairwise_knn(points, k=2)
    # print(dists, idx)
    assert dists.shape == (3, 2)
    assert idx.shape == (3, 2)
    # nearest neighbor of (0,0,0) should be (1,0,0) or (0,1,0)
    assert idx[0,1].item() in [1,2]


def test_extract_boundary_points_square():
    # simple square segmentation in 2D
    coords = torch.tensor([[x,y,0.0] for x in range(5) for y in range(5)], dtype=torch.float32)
    seg = torch.zeros(25, dtype=torch.long)
    seg[6:9] = 1  # make a small block
    sample_id = torch.zeros(25, dtype=torch.long)

    boundary_idx, boundary_coords, mean_dist = extract_boundary_points(seg, coords, sample_id, k=3)
    assert isinstance(boundary_idx, torch.Tensor)
    assert isinstance(boundary_coords, torch.Tensor)
    assert mean_dist > 0
    # boundary points must be fewer than all points
    assert len(boundary_idx) < len(seg)


def test_extract_boundary_points_equivalence():
    """
    Verify that the optimized extract_boundary_points implementation produces
    exactly the same output indices, coordinates, and mean neighbor distances
    as the original method.
    """
    from nimosef.losses.utils import extract_boundary_points as new_extract
    from pytorch3d.ops import knn_points

    # Original implementation copied for baseline comparison
    def old_extract(segmentation, coords, sample_id, k=5, max_boundary_points=20000):
        boundary_indices_list = []
        boundary_coords_list = []
        mean_distances = []
        
        unique_samples = torch.unique(sample_id)
        per_subject_cap = max_boundary_points // max(len(unique_samples), 1)
        
        for subj in unique_samples:
            subj_mask = (sample_id == subj)
            subj_indices = subj_mask.nonzero(as_tuple=True)[0]
            if len(subj_indices) <= k:
                continue
            if len(subj_indices) > per_subject_cap:
                perm = torch.randperm(len(subj_indices), device=subj_indices.device)[:per_subject_cap]
                subj_indices = subj_indices[perm]
            subj_coords = coords[subj_indices]
            subj_seg = segmentation[subj_indices]
            coords_batch = subj_coords.unsqueeze(0).float()
            knn = knn_points(coords_batch, coords_batch, K=k + 1)
            knn_indices = knn.idx[0, :, 1:]
            seg_expanded = subj_seg.unsqueeze(1).expand(-1, k)
            neighbor_seg = seg_expanded.gather(0, knn_indices)
            is_boundary = (seg_expanded != neighbor_seg).any(dim=1)
            b_idx_in_subj = is_boundary.nonzero(as_tuple=True)[0]
            if len(b_idx_in_subj) > 0:
                actual_b_idx = subj_indices[b_idx_in_subj]
                boundary_indices_list.append(actual_b_idx)
                boundary_coords_list.append(subj_coords[b_idx_in_subj])
                mean_distances.append((subj_coords[knn_indices] - subj_coords.unsqueeze(1)).norm(dim=2).mean())

        if len(boundary_indices_list) == 0:
            return torch.tensor([], dtype=torch.long, device=coords.device), torch.empty((0, coords.shape[1]), device=coords.device), torch.tensor(0.0, device=coords.device)

        boundary_indices = torch.cat(boundary_indices_list)
        boundary_coords = torch.cat(boundary_coords_list)
        mean_distance = torch.stack(mean_distances).mean()
        return boundary_indices.long(), boundary_coords, mean_distance

    # Set seed and create inputs
    torch.manual_seed(42)
    N = 100
    coords = torch.rand(N, 3)
    segmentation = torch.randint(0, 4, (N,))
    sample_id = torch.randint(0, 3, (N,))

    # We must ensure they use the same permutation if any subsampling happens,
    # so we set seeds before each run or use a large enough budget so no permutation is triggered.
    torch.manual_seed(123)
    idx_old, coords_old, dist_old = old_extract(segmentation, coords, sample_id, k=5, max_boundary_points=1000)
    
    torch.manual_seed(123)
    idx_new, coords_new, dist_new = new_extract(segmentation, coords, sample_id, k=5, max_boundary_points=1000)

    assert torch.equal(idx_old, idx_new)
    assert torch.allclose(coords_old, coords_new)
    assert torch.allclose(dist_old, dist_new, atol=1e-5)
