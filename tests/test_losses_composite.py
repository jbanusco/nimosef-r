import torch
from nimosef.losses.composite import CompositeLoss


def make_fake_preds_targets(N=20, C=4, requires_grad=False):
    coords = torch.rand(N, 3)
    sample_id = torch.zeros(N, dtype=torch.long)
    resolution = torch.tensor([1.0, 1.0, 1.0])

    preds_t0 = {
        "intensity_pred": torch.rand(N, 1, requires_grad=requires_grad),
        "seg_pred": torch.rand(N, C, requires_grad=requires_grad),
        "displacement": torch.zeros(N, 3, requires_grad=requires_grad),
        "h": torch.rand(N, 8, requires_grad=requires_grad),
    }
    preds_t = {
        "intensity_pred": torch.rand(N, 1, requires_grad=requires_grad),
        "seg_pred": torch.rand(N, C, requires_grad=requires_grad),
        "displacement": torch.zeros(N, 3, requires_grad=requires_grad),
        "h": torch.rand(N, 8, requires_grad=requires_grad),
    }
    targets_t0 = {"intensity": torch.rand(N), "segmentation": torch.randint(0, C, (N,))}
    targets_t = {"intensity": torch.rand(N), "segmentation": torch.randint(0, C, (N,))}
    return sample_id, coords, resolution, preds_t0, preds_t, targets_t0, targets_t


def test_composite_loss_forward_and_keys():
    comp = CompositeLoss()
    args = make_fake_preds_targets(N=20, requires_grad=False)
    total, loss_dict = comp(*args)
    print(total, loss_dict)
    assert isinstance(total, torch.Tensor)
    assert all(k in loss_dict for k in ["L_intensity", "L_seg", "L_disp"])


def test_composite_loss_zero_displacement_gives_low_disp():
    comp = CompositeLoss()
    args = make_fake_preds_targets(N=20, requires_grad=False)
    preds_t0, preds_t = args[3], args[4]
    preds_t0["displacement"] = torch.zeros_like(preds_t0["displacement"])
    preds_t["displacement"] = torch.zeros_like(preds_t["displacement"])
    args = args[:3] + (preds_t0, preds_t) + args[5:]
    total, loss_dict = comp(*args)
    assert loss_dict["L_disp_reg"] < 1e-4  # more tolerant


def test_composite_loss_backward():
    comp = CompositeLoss()
    args = make_fake_preds_targets(N=20, requires_grad=True)
    total, _ = comp(*args)
    total.backward()
    for key in ["seg_pred", "intensity_pred", "displacement"]:
        # print(args[3][key].grad)
        assert args[3][key].grad is not None
        assert args[4][key].grad is not None
    
    assert args[3]['h'].grad is None  # pred at t0 does not need grad on h
    assert args[4]['h'].grad is not None  # pred at t needs grad on h


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


def test_composite_loss_numerical_grad_intensity():
    comp = CompositeLoss(hp_dict={"num_labels": 4})
    args = make_fake_preds_targets(N=10, C=4, requires_grad=True)
    sample_id, coords, resolution, preds_t0, preds_t, targets_t0, targets_t = args

    intensity_pred = preds_t0["intensity_pred"]

    def loss_fn(x):
        preds_t0_mod = dict(preds_t0)
        preds_t0_mod["intensity_pred"] = x
        total, _ = comp(sample_id, coords, resolution, preds_t0_mod, preds_t, targets_t0, targets_t)
        return total

    total, _ = comp(sample_id, coords, resolution, preds_t0, preds_t, targets_t0, targets_t)
    total.backward()
    autograd_grad = intensity_pred.grad.clone()

    fd_grad = numerical_grad(loss_fn, intensity_pred.detach().clone())
    diff = (autograd_grad - fd_grad).abs().mean().item()
    assert diff < 1e-2, f"Gradient mismatch too large: {diff}"
