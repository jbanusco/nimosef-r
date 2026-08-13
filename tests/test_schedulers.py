import torch
from nimosef.training.schedulers import ClampedStepLR


def test_clamped_step_lr():
    model = torch.nn.Linear(2,2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = ClampedStepLR(opt, step_size=1, gamma=0.1, min_lrs=[0.05])
    opt.step()
    sched.step()
    assert opt.param_groups[0]['lr'] >= 0.05
