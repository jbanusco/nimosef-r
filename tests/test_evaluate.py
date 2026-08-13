import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nimosef.analysis.evaluate_network as evaluate


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        rwc = torch.stack(torch.meshgrid(
            torch.linspace(0, 2, 3),
            torch.linspace(0, 2, 3),
            torch.linspace(0, 2, 3),
            indexing="ij"
        ), dim=-1).view(-1, 3).float()
        t0_data = (torch.zeros(rwc.shape[0]), torch.zeros(rwc.shape[0], dtype=torch.long), torch.zeros(rwc.shape[0]))
        tgt_data = (torch.zeros(rwc.shape[0]), torch.zeros(rwc.shape[0], dtype=torch.long), torch.zeros(rwc.shape[0]))
        return rwc, t0_data, tgt_data, 0.0, {}, torch.zeros(rwc.shape[0], dtype=torch.long)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # add one trivial parameter so next(model.parameters()) works
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, coords, t, sample_id):
        N = coords.shape[0]
        return (
            torch.softmax(torch.rand(N, 4), dim=1),
            torch.rand(N, 1),
            torch.zeros(N, 3),
            torch.rand(N, 8)
        )


def test_plot_images_creates_file(tmp_path):
    dataset = DummyDataset()
    model = DummyModel()

    save_folder = tmp_path
    evaluate.plot_images(model, dataset, 0, save_folder)

    out_file = os.path.join(save_folder, "pred_image.png")
    assert os.path.exists(out_file)
    img = plt.imread(out_file)
    assert img.ndim in (2, 3)  # grayscale or RGB
