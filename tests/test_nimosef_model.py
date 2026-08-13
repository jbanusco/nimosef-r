import torch
import pytest
from nimosef.models.layers import INR
from nimosef.models.nimosef import MultiHeadNetwork


def test_inr_forward_shapes_residual_and_nonresidual():
    torch.manual_seed(0)
    coord_size, embed_size, hidden_size, out_size = 3, 4, 8, 6
    N = 10
    coords = torch.rand(N, coord_size)
    embed = torch.rand(N, embed_size)

    # With residual
    inr = INR(coord_size, embed_size, embed_size, embed_size, num_hidden_layers=2, use_residual=True)
    out_res = inr((coords, embed))
    assert out_res.shape == (N, embed_size)

    # Without residual
    inr = INR(coord_size, embed_size, hidden_size, out_size, num_hidden_layers=2, use_residual=False)
    out_nores = inr((coords, embed))
    assert out_nores.shape == (N, out_size)


def test_inr_backward():
    torch.manual_seed(0)
    coord_size, embed_size, out_size = 3, 4, 6
    N = 5
    coords = torch.rand(N, coord_size, requires_grad=True)
    embed = torch.rand(N, embed_size, requires_grad=True)

    input_size = coord_size + 1
    hidden_size = embed_size
    inr = INR(coord_size, embed_size, embed_size, embed_size, num_hidden_layers=2, use_residual=True)
    #NOTE: if use residual the inner size has to be the same as embed_size
    out = inr((coords, embed))

    loss = out.mean()
    loss.backward()

    assert coords.grad is not None
    assert embed.grad is not None


def test_multiheadnetwork_forward_shapes():
    torch.manual_seed(0)
    num_subjects, num_labels, latent_size, motion_size = 2, 4, 8, 6
    hidden_size = latent_size  # For the residual
    subject_slices = {i: 10 for i in range(num_subjects)}
    model = MultiHeadNetwork(
        subject_slices=subject_slices,
        num_subjects=num_subjects,
        num_labels=num_labels,
        latent_size=latent_size,
        motion_size=motion_size,
        hidden_size=hidden_size,
        num_res_layers=2,
        linear_head=True,
        skip_correction=False,
    )

    N = 20
    coords = torch.rand(N, 3)
    time = torch.rand(N, 1)
    sample_idx = torch.randint(0, num_subjects, (N,))
    slice_indices = torch.randint(0, 10, (N,))

    seg_pred, intensity_pred, displacement, h = model(coords, time, sample_idx, slice_indices)

    assert seg_pred.shape == (N, num_labels)
    assert torch.allclose(torch.softmax(seg_pred, dim=1).sum(dim=1), torch.ones(N), atol=1e-5), "Segmentation softmax must sum to 1"
    assert intensity_pred.shape == (N, 1)
    assert (0 <= intensity_pred).all() and (intensity_pred <= 1).all(), "Intensity must be in [0,1]"
    assert displacement.shape == (N, 3)
    assert h.shape[1] == latent_size


def test_multiheadnetwork_decode_latent():
    torch.manual_seed(0)
    num_subjects, num_labels, latent_size, motion_size = 3, 4, 8, 6
    hidden_size = latent_size  # For the residual
    subject_slices = {i: 10 for i in range(num_subjects)}
    model = MultiHeadNetwork(
        subject_slices, num_subjects, num_labels, latent_size, motion_size, hidden_size=hidden_size
    )

    N = 10
    coords = torch.rand(N, 3)
    time = torch.rand(N, 1)
    h = torch.rand(N, latent_size)

    latent_t = model.decode_latent(coords, time, h)
    assert latent_t.shape == (N, latent_size)


def test_multiheadnetwork_backward_pass():
    torch.manual_seed(0)
    num_subjects, num_labels, latent_size, motion_size = 2, 3, 8, 6
    hidden_size = latent_size  # For the residual
    subject_slices = {i: 10 for i in range(num_subjects)}
    model = MultiHeadNetwork(
        subject_slices, num_subjects, num_labels, latent_size, motion_size,
        hidden_size=hidden_size, skip_correction=False
    )

    N = 15
    coords = torch.rand(N, 3, requires_grad=True)
    time = torch.rand(N, 1, requires_grad=True)
    sample_idx = torch.randint(0, num_subjects, (N,))
    slice_indices = torch.randint(0, 10, (N,))

    seg_pred, intensity_pred, displacement, h = model(coords, time, sample_idx, slice_indices)
    loss = seg_pred.mean() + intensity_pred.mean() + displacement.mean() + h.mean()
    loss.backward()

    # Check gradients flow
    assert coords.grad is not None
    assert time.grad is not None
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def numerical_grad_scalar(f, x, eps=1e-4):
    """Finite-difference gradient for scalar f(x) wrt tensor x."""
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


# ---------------------------
# INR
# ---------------------------
def test_inr_numerical_grad():
    torch.manual_seed(0)
    coord_size, embed_size = 3, 4
    N = 5
    coords = torch.rand(N, coord_size, requires_grad=True)
    embed = torch.rand(N, embed_size, requires_grad=True)

    inr = INR(coord_size, embed_size, embed_size, embed_size,
              num_hidden_layers=2, use_residual=True)

    def loss_fn(c, e):
        return inr((c, e)).sum()

    # autograd
    out = inr((coords, embed))
    out.sum().backward()
    autograd_grad_coords = coords.grad.clone()
    autograd_grad_embed = embed.grad.clone()

    # finite difference (coords)
    fd_grad_coords = numerical_grad_scalar(lambda c: loss_fn(c, embed.detach()), coords.detach().clone())
    # finite difference (embed)
    fd_grad_embed = numerical_grad_scalar(lambda e: loss_fn(coords.detach(), e), embed.detach().clone())

    diff_coords = (autograd_grad_coords - fd_grad_coords).abs().mean().item()
    diff_embed = (autograd_grad_embed - fd_grad_embed).abs().mean().item()

    assert diff_coords < 1e-2, f"INR coords grad mismatch: {diff_coords}"
    assert diff_embed < 1e-2, f"INR embed grad mismatch: {diff_embed}"


# ---------------------------
# MultiHeadNetwork
# ---------------------------
def test_multiheadnetwork_numerical_grad():
    torch.manual_seed(0)
    num_subjects, num_labels, latent_size, motion_size = 2, 3, 4, 2
    hidden_size = latent_size
    subject_slices = {i: 10 for i in range(num_subjects)}
    model = MultiHeadNetwork(
        subject_slices, num_subjects, num_labels, latent_size, motion_size,
        hidden_size=hidden_size, skip_correction=False
    )

    N = 6
    coords = torch.rand(N, 3, requires_grad=True)
    time = torch.rand(N, 1, requires_grad=True)
    sample_idx = torch.randint(0, num_subjects, (N,))
    slice_indices = torch.randint(0, 10, (N,))

    def loss_fn(c, t):
        seg_pred, intensity_pred, displacement, h = model(c, t, sample_idx, slice_indices)
        return seg_pred.sum() + intensity_pred.sum() + displacement.sum() + h.sum()

    # autograd
    out = loss_fn(coords, time)
    out.backward()
    autograd_grad_coords = coords.grad.clone()
    autograd_grad_time = time.grad.clone()

    # finite difference (coords)
    fd_grad_coords = numerical_grad_scalar(lambda c: loss_fn(c, time.detach()), coords.detach().clone())
    # finite difference (time)
    fd_grad_time = numerical_grad_scalar(lambda t: loss_fn(coords.detach(), t), time.detach().clone())

    diff_coords = (autograd_grad_coords - fd_grad_coords).abs().mean().item()
    diff_time = (autograd_grad_time - fd_grad_time).abs().mean().item()

    assert diff_coords < 1e-2, f"MultiHead coords grad mismatch: {diff_coords}"
    assert diff_time < 1e-2, f"MultiHead time grad mismatch: {diff_time}"


def test_multiheadnetwork_numerical_grad_embeddings():
    torch.manual_seed(0)
    num_subjects, num_labels, latent_size, motion_size = 3, 3, 4, 2
    hidden_size = latent_size
    subject_slices = {i: 10 for i in range(num_subjects)}
    model = MultiHeadNetwork(
        subject_slices, num_subjects, num_labels, latent_size, motion_size,
        hidden_size=hidden_size, skip_correction=False
    )

    N = 4
    coords = torch.rand(N, 3)
    time = torch.rand(N, 1)
    sample_idx = torch.randint(0, num_subjects, (N,))
    slice_indices = torch.randint(0, 10, (N,))

    def loss_fn():
        seg_pred, intensity_pred, displacement, h = model(coords, time, sample_idx, slice_indices)
        return seg_pred.sum() + intensity_pred.sum() + displacement.sum() + h.sum()

    # autograd
    loss = loss_fn()
    loss.backward()

    autograd_grad_shape = model.shape_code.weight.grad.clone()
    autograd_grad_sc = model.slice_correction.subject_tables["0"].weight.grad.clone()

    # finite difference (shape_code weights)
    shape_weight = model.shape_code.weight.data.clone()
    fd_grad_shape = torch.zeros_like(shape_weight)

    eps = 1e-4
    for i in range(shape_weight.numel()):
        orig = shape_weight.view(-1)[i].item()
        model.shape_code.weight.data.view(-1)[i] = orig + eps
        f_pos = loss_fn().item()
        model.shape_code.weight.data.view(-1)[i] = orig - eps
        f_neg = loss_fn().item()
        model.shape_code.weight.data.view(-1)[i] = orig
        fd_grad_shape.view(-1)[i] = (f_pos - f_neg) / (2 * eps)

    # finite difference (slice correction weights for subject 0)
    sc_weight = model.slice_correction.subject_tables["0"].weight.data.clone()
    fd_grad_sc = torch.zeros_like(sc_weight)

    for i in range(sc_weight.numel()):
        orig = sc_weight.view(-1)[i].item()
        model.slice_correction.subject_tables["0"].weight.data.view(-1)[i] = orig + eps
        f_pos = loss_fn().item()
        model.slice_correction.subject_tables["0"].weight.data.view(-1)[i] = orig - eps
        f_neg = loss_fn().item()
        model.slice_correction.subject_tables["0"].weight.data.view(-1)[i] = orig
        fd_grad_sc.view(-1)[i] = (f_pos - f_neg) / (2 * eps)

    # Compare
    diff_shape = (autograd_grad_shape - fd_grad_shape).abs().mean().item()
    diff_sc = (autograd_grad_sc - fd_grad_sc).abs().mean().item()

    assert diff_shape < 1e-2, f"Shape embedding gradient mismatch too large: {diff_shape}"
    assert diff_sc < 1e-2, f"Slice correction embedding gradient mismatch too large: {diff_sc}"
