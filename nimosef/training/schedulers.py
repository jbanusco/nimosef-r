import torch


class ClampedStepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, min_lrs=None, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.
            min_lrs (list or None): A list of minimum learning rates for each parameter group.
                                    If None, defaults to 0 for all groups.
        """     
        if min_lrs is None:
            self.min_lrs = [0.0] * len(optimizer.param_groups)
        else:
            self.min_lrs = min_lrs
        super().__init__(optimizer, step_size, gamma, last_epoch)   

    def get_lr(self):
        # Compute the usual new LRs using StepLR's logic
        lrs = super().get_lr()
        # Clamp each learning rate to its specified minimum
        clamped_lrs = [max(new_lr, min_lr) for new_lr, min_lr in zip(lrs, self.min_lrs)]
        return clamped_lrs