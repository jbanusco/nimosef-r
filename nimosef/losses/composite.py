import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from nimosef.losses.base import dice_loss, compute_graph_connectivity_loss, approx_jacobian_loss, approx_laplacian_on_graph
from nimosef.losses.utils import extract_boundary_points


class CompositeLoss(nn.Module):
    """
    Composite loss combining intensity, segmentation, latent reg, displacement, connectivity, smoothness.
    """

    def __init__(self, is_test=False, device="cpu", hp_dict=None, manifold=None):
        super().__init__()
        self.device = device
        self.is_test = is_test
        self.manifold = manifold

        defaults = {
            "num_labels": 4,
            "lambda_rec": 1.0,
            "lambda_seg": 1.0,
            "lambda_reg": 1e-4,
            "lambda_dsp": 0.8,
            "lambda_reg_dsp": 1.0,
            "lambda_jacobian": 1.0,
            "lambda_vol": 0.1,
            "lambda_graph_conn": 1.0,
            "lambda_smoothness": 1.0,
            "warmup_epochs": 0,
            "tgt_lambda": 0.1,  # for the connectivity loss
            "k_neighs": 5,
        }
        if hp_dict is not None:
            defaults.update(hp_dict)
        self.__dict__.update(defaults)

        self.rec_loss = nn.L1Loss(reduction="none")
        # self.rec_loss = nn.HuberLoss(reduction="none", delta=0.05)
        # self.rec_loss = nn.SmoothL1Loss(reduction="none", beta=0.05)
        class_weights = torch.tensor([0.3, 1.0, 1.0, 1.0]).to(device)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    def forward(self, sample_id, coords, resolution, preds_t0, preds_t, targets_t0, targets_t, epoch=0):
        # Compute warmup factor (just in case we want to use it later)
        warmup_factor = min(1.0, epoch / self.warmup_epochs) if self.warmup_epochs else 1.0

        # ================================
        # 1. Intensity
        pred_int_t0 = preds_t0["intensity_pred"][..., 0]
        pred_int_t = preds_t["intensity_pred"][..., 0]
        L_intensity = self.rec_loss(pred_int_t0, targets_t0["intensity"]).mean()
        L_intensity += self.rec_loss(pred_int_t, targets_t["intensity"]).mean()

        # ================================
        # 2. Segmentation Loss (Dice + CE)
        if self.is_test:
            with torch.no_grad():
                # Dice
                L_seg = dice_loss(preds_t0["seg_pred"], targets_t0["segmentation"], self.num_labels, ignore_background=True, weighted=False, apply_softmax=True)[1:].mean()
                L_seg += dice_loss(preds_t["seg_pred"], targets_t["segmentation"], self.num_labels, ignore_background=True, weighted=False, apply_softmax=True)[1:].mean()
            
                # Cross-entropy
                L_seg += self.ce_loss(preds_t0["seg_pred"], targets_t0["segmentation"]).mean()
                L_seg += self.ce_loss(preds_t["seg_pred"], targets_t["segmentation"]).mean()
        else:
            # Dice
            L_seg = dice_loss(preds_t0["seg_pred"], targets_t0["segmentation"], self.num_labels, ignore_background=True, weighted=False, apply_softmax=True)[1:].mean()
            L_seg += dice_loss(preds_t["seg_pred"], targets_t["segmentation"], self.num_labels, ignore_background=True, weighted=False, apply_softmax=True)[1:].mean()

            # Cross-entropy
            L_seg += self.ce_loss(preds_t0["seg_pred"], targets_t0["segmentation"]).mean()
            L_seg += self.ce_loss(preds_t["seg_pred"], targets_t["segmentation"]).mean()

        # ================================
        # 3. Latent Regularization Loss
        if self.manifold is not None:
            # Use hyperbolic/spherical distance to origin squared
            h = preds_t["h"]
            if hasattr(self.manifold, "c"): # Poincare Ball check
                # Numerical safety: Poincare ball is undefined at norm >= 1
                eps = 1e-5
                norm = h.norm(dim=-1, keepdim=True)
                h = h * torch.min(torch.ones_like(norm), (1 - eps) / norm.clamp(min=eps))
            
            L_latent = self.manifold.dist0(h).pow(2).mean()
        else:
            L_latent = preds_t["h"].square().mean(dim=0).sum()        

        # ================================
        # 4. Displacement Loss: Chamfer distance on boundaries
        seg_t0_labels = torch.argmax(preds_t0["seg_pred"], dim=1)
        seg_t_labels = torch.argmax(preds_t["seg_pred"], dim=1)
        if self.is_test:
            with torch.no_grad():
                # ---- Use the predicted segmentation boundary points ----
                b_indices_t0, boundary_coords_t0, sq_mean_distance_t0 = extract_boundary_points(seg_t0_labels, coords, sample_id, k=self.k_neighs)
                b_indices_t, boundary_coords_t, sq_mean_distance_t = extract_boundary_points(seg_t_labels, coords, sample_id, k=self.k_neighs)
        else:
            # ---- Use the target segmentation boundary points ----
            b_indices_t0, boundary_coords_t0, sq_mean_distance_t0 = extract_boundary_points(targets_t0["segmentation"], coords, sample_id, k=self.k_neighs)
            b_indices_t, boundary_coords_t, sq_mean_distance_t = extract_boundary_points(targets_t["segmentation"], coords, sample_id, k=self.k_neighs)

        # For the boundary points from t0, sample the predicted displacement at time t.
        displacement_t0 = preds_t0["displacement"]
        displacement_t = preds_t["displacement"]

        # Predicted displacement at boundary points at time t0
        pred_disp_at_boundary = displacement_t[b_indices_t0]

        # Predicted boundary at target time t: shift the t0 boundary by the predicted displacement.
        predicted_boundary_t = boundary_coords_t0 + pred_disp_at_boundary
        
        # Boundary tracking loss: compute the Chamfer distance between the predicted boundary and the target boundary.
        chamfer_loss = chamfer_distance(
            predicted_boundary_t.unsqueeze(0),
            boundary_coords_t.unsqueeze(0),
            norm=2,
            single_directional=True,
            batch_reduction=None,
            point_reduction="mean",
        )[0]  # 'max' = Hausdorff, 'mean' = Chamfer


        # TOCHECK: Scale by the original resolution? s.t. the displacement is in [voxels]
        # L_disp = chamfer_loss #/ min_res.detach()  # Normalize by the minimum resolution        
        mean_distance_t0 = torch.sqrt(sq_mean_distance_t0) #.detach()
        distance_factor = torch.tensor([0.01], device=sq_mean_distance_t0.device).clone()
        inv_distance_factor = 1.0 / (distance_factor  * 1)

        # Normalize by the mean distance of the boundary points, to have an idea of the scale of the displacement
        L_disp = chamfer_loss * inv_distance_factor * warmup_factor 

        # ================================
        # 5. Displacement regularization loss
        L_disp_reg = preds_t0["displacement"].square().mean() * inv_distance_factor
        L_disp_reg *= warmup_factor

        # ================================
        # 6. Graph Connectivity and Jacobian loss
        # Scale the boundary coordinates to the original resolution?        
        sc_boundary_coords_t0 = boundary_coords_t0 #/ resolution
        sc_boundary_coords_tgt = boundary_coords_t #/ resolution

        # graph_max_distance = sq_mean_distance_t0 * 1.5
        graph_max_distance = distance_factor * 2
        # graph_max_distance = self.max_knn_distance
        
        L_graph_t0, g_t0 = compute_graph_connectivity_loss(
            sample_id[b_indices_t0],
            sc_boundary_coords_t0,
            k=self.k_neighs,
            target_lambda=self.tgt_lambda,
            max_knn_distance=graph_max_distance,
        )
        L_graph_t, g_t = compute_graph_connectivity_loss(
            sample_id[b_indices_t],
            sc_boundary_coords_tgt,
            k=self.k_neighs,
            target_lambda=self.tgt_lambda,
            max_knn_distance=graph_max_distance,
        )
        L_graph = L_graph_t0 + L_graph_t

        if g_t is None or g_t0 is None:
            print("No valid graph edges; skipping connectivity loss.")
            L_Jfold, L_Jvol, L_smooth_int, L_smooth_seg = (torch.tensor(0.0).to(self.device),) * 4
            L_J = L_Jfold + L_Jvol * self.lambda_vol
            L_smooth = L_smooth_int + L_smooth_seg
        else:
            # Use reference (undeformed) coordinates boundary_coords_t0 rather than deformed coordinates.
            # This aligns with the Lagrangian deformation gradient F = I + du/dX.
            # References:
            # - Mok & Chung, MICCAI 2020: Fast Symmetric Diffeomorphic Image Registration
            # - Balakrishnan et al., IEEE TMI 2019: VoxelMorph: A Learning Framework for Deformable Medical Image Registration
            # - Beg et al., IJCV 2005: Computing Large Deformation Metric Mappings via Geodesic Flows
            L_Jfold, L_Jvol = approx_jacobian_loss(displacement_t[b_indices_t0], boundary_coords_t0, g_t0)
            L_J = (L_Jfold + L_Jvol * self.lambda_vol) * warmup_factor
            # ================================
            # 7. Smoothness Loss: Laplacian on the graph
            # --- Build masks to avoid cross-class edges ---
            src0, dst0 = g_t0.edges()
            src1, dst1 = g_t.edges()
            if not self.is_test:
                labels0 = targets_t0["segmentation"][b_indices_t0]
                labels1 = targets_t["segmentation"][b_indices_t]
            else:
                labels0 = seg_t0_labels[b_indices_t0]
                labels1 = seg_t_labels[b_indices_t]
            same0 = (labels0[src0] == labels0[dst0])
            same1 = (labels1[src1] == labels1[dst1])
                        
            # Normalised by edge
            normalize_by_length = False

            # Compute smoothness of segmentation and intensity
            boundary_intensity_t0 = preds_t0["intensity_pred"][...,0][b_indices_t0]
            boundary_intensity_t = preds_t["intensity_pred"][...,0][b_indices_t]
            L_smooth_int = approx_laplacian_on_graph(boundary_intensity_t0, boundary_coords_t0, g_t0, edge_mask=same0, normalize_by_length=normalize_by_length)
            L_smooth_int += approx_laplacian_on_graph(boundary_intensity_t, boundary_coords_t, g_t, edge_mask=same1, normalize_by_length=normalize_by_length)

            # boundary_segmentation_t0 = seg_t0_labels[b_indices_t0]
            # boundary_segmentation_t = seg_t_labels[b_indices_t]
            # Use prob. instead of argmax labels
            boundary_segmentation_t0 = F.softmax(preds_t0["seg_pred"], dim=1)[b_indices_t0]  # (Nb0, C)
            boundary_segmentation_t  = F.softmax(preds_t["seg_pred"],  dim=1)[b_indices_t]   # (Nb1, C)
            L_smooth_seg = approx_laplacian_on_graph(boundary_segmentation_t0, boundary_coords_t0, g_t0, edge_mask=same0, normalize_by_length=normalize_by_length)
            L_smooth_seg += approx_laplacian_on_graph(boundary_segmentation_t, boundary_coords_t, g_t, edge_mask=same1, normalize_by_length=normalize_by_length)

            L_smooth = L_smooth_int + L_smooth_seg
            # L_smooth = torch.tensor(0.0).to(self.device)

        # =============================
        # Aggregate the loss components
        # =============================
        total = (
            self.lambda_rec * L_intensity
            + self.lambda_reg * L_latent
            + self.lambda_dsp * L_disp
            + self.lambda_reg_dsp * L_disp_reg
            + self.lambda_jacobian * L_J
            + self.lambda_graph_conn * L_graph
            + self.lambda_smoothness * L_smooth
        )

        if not self.is_test:
            # Include the segmentation loss
            total += self.lambda_seg * L_seg

        loss_components = {
            "L_intensity": L_intensity.detach().item(),
            "L_seg": L_seg.detach().item(),
            "L_latent": L_latent.detach().item(),
            "L_disp": L_disp.detach().item(),
            "L_disp_reg": L_disp_reg.detach().item(),
            "L_J": L_J.detach().item(),
            "L_Jfold": L_Jfold.detach().item(),
            "L_Jvol": L_Jvol.detach().item(),
            "L_graph": L_graph.detach().item(),
            "L_smooth_int": L_smooth_int.detach().item(),
            "L_smooth_seg": L_smooth_seg.detach().item(),
        }

        # EXPLICITLY FREE DGL GRAPHS
        if 'g_t0' in locals() and g_t0 is not None:
            del g_t0
        if 'g_t' in locals() and g_t is not None:
            del g_t

        return total, loss_components
