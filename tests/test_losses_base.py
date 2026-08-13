import torch
import dgl
import pytest
from itertools import combinations
from nimosef.losses.base import dice_loss, approx_jacobian_loss

# ==========================
# HELPER
# ==========================

def numerical_grad(f, x, eps=1e-4):
    grad = torch.zeros_like(x)
    flat_x = x.view(-1)
    flat_grad = grad.view(-1)

    for i in range(flat_x.numel()):
        orig = flat_x[i].item()
        flat_x[i] = orig + eps
        f_pos = f(x).item()
        flat_x[i] = orig - eps
        f_neg = f(x).item()
        flat_x[i] = orig
        flat_grad[i] = (f_pos - f_neg) / (2 * eps)

    return grad


def numerical_grad_scalar(f, x, eps=1e-4):
    """Finite-difference gradient for scalar-valued function."""
    grad = torch.zeros_like(x)
    flat_x = x.view(-1)
    flat_grad = grad.view(-1)

    for i in range(flat_x.numel()):
        orig = flat_x[i].item()
        flat_x[i] = orig + eps
        f_pos = f(x).item()
        flat_x[i] = orig - eps
        f_neg = f(x).item()
        flat_x[i] = orig
        flat_grad[i] = (f_pos - f_neg) / (2 * eps)

    return grad

# ==========================
# DICE
# ==========================

def test_dice_loss_perfect_prediction():
    N, C = 6, 3
    target = torch.randint(0, C, (N,))
    pred = torch.nn.functional.one_hot(target, num_classes=C).float()
    loss = dice_loss(pred, target, num_classes=C, weighted=False, apply_softmax=False)
    # Perfect match → Dice loss = 0
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)

@pytest.mark.parametrize("N,C", [(10, 3), (20, 4)])
def test_dice_loss_random_predictions(N, C):
    pred = torch.rand(N, C)
    target = torch.randint(0, C, (N,))
    loss = dice_loss(pred, target, num_classes=C)
    assert torch.all(loss >= 0)
    assert loss.shape == (C,)


def test_dice_loss_numerical_grad():
    torch.manual_seed(0)
    N, C = 6, 3
    target = torch.randint(0, C, (N,))
    pred = torch.rand(N, C, requires_grad=True)  # soft predictions

    def loss_fn(p):
        return dice_loss(p, target, num_classes=C, weighted=False).mean()

    # Autograd gradient
    loss = loss_fn(pred)
    loss.backward()
    autograd_grad = pred.grad.clone()

    # Finite-difference gradient
    fd_grad = numerical_grad(loss_fn, pred.detach().clone())

    # Compare
    diff = (autograd_grad - fd_grad).abs().mean().item()
    assert diff < 1e-2, f"Dice loss gradient mismatch too large: {diff}"


# ==========================
# Jacobian
# ==========================

def test_jacobian_loss_translation():
    N = 10
    coords = torch.rand(N, 3)
    displacement = torch.zeros_like(coords) + 0.1  # pure translation
    src = torch.arange(N-1); dst = torch.arange(1, N)
    g = dgl.graph((src, dst), num_nodes=N)

    L_fold, L_vol = approx_jacobian_loss(displacement, coords, g)
    print(L_fold, L_vol)
    assert torch.allclose(L_fold, torch.tensor(0.0), atol=1e-6)
    assert L_vol < 1e-3  # near 0, volume preserved


def test_jacobian_loss_folding():
    N = 4
    coords = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    displacement = torch.tensor([[0.,0.,0.],[-2.,0.,0.],[0.,0.,0.],[0.,0.,0.]])  # flip
    src = torch.tensor([0,0,0]); dst = torch.tensor([1,2,3])
    g = dgl.graph((src, dst), num_nodes=N)

    L_fold, L_vol = approx_jacobian_loss(displacement, coords, g)
    assert L_fold > 0  # folding detected


@pytest.mark.parametrize("factor", [0.1, 1.0, 2.0, 4.0])
def test_jacobian_loss_scaling(factor):
    N = 5
    coords = torch.rand(N, 3)
    displacement = coords * factor  # isotropic scaling (det != 1, no folding)

    # Fully connected undirected graph (all pairs)
    src, dst = zip(*[(i, j) for i, j in combinations(range(N), 2)])
    g = dgl.graph((list(src), list(dst)), num_nodes=N)

    L_fold, L_vol = approx_jacobian_loss(displacement, coords, g, penalize_volume=True)
    assert torch.allclose(L_fold, torch.tensor(0.0), atol=1e-6)
    assert L_vol > 0  # volume change penalized


def test_approx_jacobian_loss_grad():
    torch.manual_seed(0)
    N = 6
    coords = torch.rand(N, 3, requires_grad=False)
    displacement = torch.rand(N, 3, requires_grad=True)
    src = torch.arange(N-1)
    dst = torch.arange(1, N)
    g = dgl.graph((src, dst), num_nodes=N)

    def loss_fn(disp):
        L_fold, L_vol = approx_jacobian_loss(disp, coords, g)
        return (L_fold + L_vol)  # scalar total

    # Autograd gradient
    loss = loss_fn(displacement)
    loss.backward()
    autograd_grad = displacement.grad.clone()

    # Finite-difference gradient
    fd_grad = numerical_grad_scalar(loss_fn, displacement.detach().clone())

    diff = (autograd_grad - fd_grad).abs().mean().item()
    assert diff < 1e-2, f"Jacobian loss gradient mismatch too large: {diff}"


def test_approx_jacobian_loss_grad_fold_only():
    torch.manual_seed(0)
    N = 6
    coords = torch.rand(N, 3)
    displacement = torch.rand(N, 3, requires_grad=True)
    src = torch.arange(N-1)
    dst = torch.arange(1, N)
    g = dgl.graph((src, dst), num_nodes=N)

    def loss_fn(disp):
        L_fold, _ = approx_jacobian_loss(disp, coords, g, penalize_volume=False)
        return L_fold

    # Autograd gradient
    loss = loss_fn(displacement)
    loss.backward()
    autograd_grad = displacement.grad.clone()

    # Finite-difference gradient
    fd_grad = numerical_grad_scalar(loss_fn, displacement.detach().clone())

    diff = (autograd_grad - fd_grad).abs().mean().item()
    assert diff < 1e-2, f"Jacobian folding gradient mismatch too large: {diff}"


def test_approx_jacobian_loss_grad_volume_only():
    torch.manual_seed(0)
    N = 6
    coords = torch.rand(N, 3)
    displacement = torch.rand(N, 3, requires_grad=True)
    src = torch.arange(N-1)
    dst = torch.arange(1, N)
    g = dgl.graph((src, dst), num_nodes=N)

    def loss_fn(disp):
        _, L_vol = approx_jacobian_loss(disp, coords, g, penalize_volume=True)
        return L_vol

    # Autograd gradient
    loss = loss_fn(displacement)
    loss.backward()
    autograd_grad = displacement.grad.clone()

    # Finite-difference gradient
    fd_grad = numerical_grad_scalar(loss_fn, displacement.detach().clone())

    diff = (autograd_grad - fd_grad).abs().mean().item()
    assert diff < 1e-2, f"Jacobian volume gradient mismatch too large: {diff}"


def test_jacobian_loss_identity_translation():
    """Pure translation → no folding, no volume change."""
    coords = torch.tensor([[0., 0., 0.],
                           [1., 0., 0.],
                           [2., 0., 0.]])
    displacement = torch.ones_like(coords) * 2.0  # translate by +2
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 2])
    g = dgl.graph((src, dst), num_nodes=3)

    L_fold, L_vol = approx_jacobian_loss(displacement, coords, g, penalize_volume=True)
    assert torch.allclose(L_fold, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(L_vol, torch.tensor(0.0), atol=1e-6)


def test_jacobian_loss_uniform_scaling():
    """Uniform scaling → no folding, volume penalty > 0."""
    coords = torch.tensor([[0., 0., 0.],
                           [1., 0., 0.],
                           [2., 0., 0.]])
    displacement = coords * 0.5  # scale by factor 1.5
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 2])
    g = dgl.graph((src, dst), num_nodes=3)

    L_fold, L_vol = approx_jacobian_loss(displacement, coords, g, penalize_volume=True)
    assert torch.allclose(L_fold, torch.tensor(0.0), atol=1e-6)
    assert L_vol > 0


def test_jacobian_loss_folding_case():
    """Edge flip (folding) → folding penalty > 0."""
    coords = torch.tensor([[0., 0., 0.],
                           [1., 0., 0.],
                           [2., 0., 0.]])
    displacement = torch.tensor([[0., 0., 0.],
                                 [-3., 0., 0.],  # flip edge direction
                                 [0., 0., 0.]])
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 2])
    g = dgl.graph((src, dst), num_nodes=3)

    L_fold, L_vol = approx_jacobian_loss(displacement, coords, g, penalize_volume=True)
    assert L_fold > 0
    # Volume penalty may or may not trigger depending on determinant, but folding must be detected.


# ==========================
# JACOBIAN: EQUIVALENCE TESTS
# ==========================

def compute_jacobian_loop(points, displacements, weights, neighs_idx, distances):
    N = points.shape[0]
    partial_derivatives = torch.zeros((N, 3, 3), device=points.device)
    disp_diffs = displacements[neighs_idx] - displacements.unsqueeze(1)
    coords_diff = points[neighs_idx] - points.unsqueeze(1)
    distances = distances + (distances == 0).float() * 1e-6
    for i in range(3):
        for j in range(3):
            weighted_derivative = torch.sum(weights * disp_diffs[..., i] * coords_diff[..., j] / distances, dim=-1)
            partial_derivatives[:, i, j] = weighted_derivative
    identity = torch.eye(3, device=points.device).expand(N, -1, -1)
    return partial_derivatives + identity


def compute_jacobian_einsum(points, displacements, weights, neighs_idx, distances):
    disp_diffs = displacements[neighs_idx] - displacements.unsqueeze(1)
    coords_diff = points[neighs_idx] - points.unsqueeze(1)
    distances = distances + (distances == 0).float() * 1e-6
    weighted_derivatives = torch.einsum("nk,nki,nkj->nij", weights, disp_diffs, coords_diff / distances[..., None])
    identity = torch.eye(3, device=points.device).expand(points.shape[0], -1, -1)
    return weighted_derivatives + identity


def test_jacobian_equivalence():
    torch.manual_seed(42)
    N, K = 10, 5
    points = torch.rand((N, 3))
    displacements = torch.rand((N, 3))
    neighs_idx = torch.randint(0, N, (N, K))
    distances = torch.rand((N, K)) + 1e-2
    weights = torch.rand((N, K))
    weights = weights / weights.sum(dim=-1, keepdim=True)
    jac_loop = compute_jacobian_loop(points, displacements, weights, neighs_idx, distances)
    jac_einsum = compute_jacobian_einsum(points, displacements, weights, neighs_idx, distances)
    assert torch.allclose(jac_loop, jac_einsum, atol=1e-4)
    det_loop = torch.linalg.det(jac_loop)
    det_einsum = torch.linalg.det(jac_einsum)
    assert torch.allclose(det_loop, det_einsum, atol=1e-4)


def compute_jacobian_explicit(points, displacements, weights, neighs_idx, distances):
    N = points.shape[0]
    distances = distances + (distances == 0).float() * 1e-6

    disp_diffs_x = displacements[neighs_idx, 0] - displacements[:, [0]]
    disp_diffs_y = displacements[neighs_idx, 1] - displacements[:, [1]]
    disp_diffs_z = displacements[neighs_idx, 2] - displacements[:, [2]]

    coords_diff_x = points[neighs_idx, 0] - points[:, [0]]
    coords_diff_y = points[neighs_idx, 1] - points[:, [1]]
    coords_diff_z = points[neighs_idx, 2] - points[:, [2]]

    partial_derivatives = torch.zeros((N, 3, 3), device=points.device)
    partial_derivatives[:, 0, 0] = torch.sum(weights * disp_diffs_x * coords_diff_x / distances, dim=-1)
    partial_derivatives[:, 0, 1] = torch.sum(weights * disp_diffs_x * coords_diff_y / distances, dim=-1)
    partial_derivatives[:, 0, 2] = torch.sum(weights * disp_diffs_x * coords_diff_z / distances, dim=-1)

    partial_derivatives[:, 1, 0] = torch.sum(weights * disp_diffs_y * coords_diff_x / distances, dim=-1)
    partial_derivatives[:, 1, 1] = torch.sum(weights * disp_diffs_y * coords_diff_y / distances, dim=-1)
    partial_derivatives[:, 1, 2] = torch.sum(weights * disp_diffs_y * coords_diff_z / distances, dim=-1)

    partial_derivatives[:, 2, 0] = torch.sum(weights * disp_diffs_z * coords_diff_x / distances, dim=-1)
    partial_derivatives[:, 2, 1] = torch.sum(weights * disp_diffs_z * coords_diff_y / distances, dim=-1)
    partial_derivatives[:, 2, 2] = torch.sum(weights * disp_diffs_z * coords_diff_z / distances, dim=-1)

    identity = torch.eye(3, device=points.device).expand(N, -1, -1)
    return partial_derivatives + identity


def test_jacobian_equivalence_all():
    torch.manual_seed(42)
    N, K = 10, 5
    points = torch.rand((N, 3))
    displacements = torch.rand((N, 3))
    neighs_idx = torch.randint(0, N, (N, K))
    distances = torch.rand((N, K)) + 1e-2
    weights = torch.rand((N, K))
    weights = weights / weights.sum(dim=-1, keepdim=True)

    jac_loop = compute_jacobian_loop(points, displacements, weights, neighs_idx, distances)
    jac_einsum = compute_jacobian_einsum(points, displacements, weights, neighs_idx, distances)
    jac_explicit = compute_jacobian_explicit(points, displacements, weights, neighs_idx, distances)

    # Equivalence check
    assert torch.allclose(jac_loop, jac_einsum, atol=1e-4)
    assert torch.allclose(jac_loop, jac_explicit, atol=1e-4)

    # Determinant check
    det_loop = torch.linalg.det(jac_loop)
    det_einsum = torch.linalg.det(jac_einsum)
    det_explicit = torch.linalg.det(jac_explicit)
    assert torch.allclose(det_loop, det_einsum, atol=1e-4)
    assert torch.allclose(det_loop, det_explicit, atol=1e-4)


def test_jacobian_determinant_lemma_equivalence():
    """
    Verify that our simplified formulation matches the exact determinant
    obtained via the standard batch linalg.det solver.
    
    Mathematical Foundation (Matrix Determinant Lemma):
    For a rank-one update of the identity matrix J = I + u v^T, the determinant is:
        det(J) = det(I + u v^T) = 1 + u^T v
    Here:
        u = du (derivative magnitude with finite differences)
        v = delta_x (unit direction vector)
    This is mathematically exact and avoids costly 3x3 determinant solvers and O(E*3*3) memory.
    
    Reference:
    https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    """
    torch.manual_seed(42)
    E = 1000
    du = torch.randn(E, 3)
    delta_x = torch.randn(E, 3)
    
    # 1. Full outer product determinant (standard method)
    J = du.unsqueeze(-1) * delta_x.unsqueeze(-2)  # (E, 3, 3)
    eye = torch.eye(3, device=du.device).expand(J.shape[0], -1, -1)
    J = J + eye
    det_standard = torch.linalg.det(J)
    
    # 2. Simplified formulation via Matrix Determinant Lemma
    det_lemma = 1.0 + (du * delta_x).sum(dim=-1)
    
    assert torch.allclose(det_standard, det_lemma, atol=1e-5, rtol=1e-5)