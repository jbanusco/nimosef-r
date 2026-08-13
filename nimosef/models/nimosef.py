import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import geoopt

from nimosef.models.layers import INR


class TimeTrigEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, coords, time):
        sin_t = torch.sin(2*math.pi*time).float()
        cos_t = torch.cos(2*math.pi*time).float()
        out = torch.cat([coords.float(), sin_t, cos_t], dim=1)
        return out

class RiemannianEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, manifold):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold = manifold
        
        # Ensure weight is created on the correct device (same as manifold)
        # Manifolds often have internal buffers or parameters (like curvature 'c')
        device = getattr(manifold, "c", torch.tensor(0.)).device if hasattr(manifold, "c") else torch.device("cpu")
        
        weight = torch.randn(num_embeddings, embedding_dim, device=device) * (0.01 / math.sqrt(embedding_dim))
        weight = self.manifold.projx(weight) # ensure it starts valid
        self.weight = geoopt.ManifoldParameter(weight, manifold=manifold)

    def forward(self, indices):
        return self.manifold.projx(self.weight[indices])


class SliceEmbeddingWeightView:
    def __init__(self, parent_weight, start, end):
        self.parent_weight = parent_weight
        self.start = start
        self.end = end

    @property
    def data(self):
        return self.parent_weight.data[self.start : self.end]

    @property
    def grad(self):
        if self.parent_weight.grad is None:
            return None
        return self.parent_weight.grad[self.start : self.end]

    def __getattr__(self, name):
        w_slice = self.parent_weight[self.start : self.end]
        return getattr(w_slice, name)

    def __getitem__(self, idx):
        return self.parent_weight[self.start : self.end][idx]

    def __setitem__(self, idx, value):
        self.parent_weight.data[self.start : self.end][idx] = value


class SliceEmbeddingView:
    def __init__(self, parent_table, subj_id, max_slices, num_slices):
        self.parent_table = parent_table
        self.subj_id = subj_id
        self.max_slices = max_slices
        self.num_slices = num_slices

    @property
    def weight(self):
        start = self.subj_id * self.max_slices
        return SliceEmbeddingWeightView(self.parent_table.weight, start, start + self.num_slices)

    def __call__(self, slices):
        flat_idx = self.subj_id * self.max_slices + slices
        return self.parent_table(flat_idx)


class SliceCorrection(nn.Module):
    """
    Per-subject, per-slice 2D affine corrections (2x2 matrix + translation)
    fully vectorized to avoid CPU-GPU sync bottlenecks.
    """
    def __init__(self, subject_slices: dict):
        super().__init__()
        self.subject_slices = subject_slices
        self.num_subjects = len(subject_slices)
        self.max_slices = max(subject_slices.values()) if subject_slices else 1
        
        self.table = nn.Embedding(self.num_subjects * self.max_slices, 3)
        with torch.no_grad():
            self.table.weight.fill_(0.0)

    def reinit(self):
        with torch.no_grad():
            self.table.weight.fill_(0.0)

    def forward(self, subject_ids, slice_indices):
        subject_ids = subject_ids.view(-1)
        slice_indices = slice_indices.view(-1)
        
        # O(1) Vectorized lookup
        flat_idx = subject_ids * self.max_slices + slice_indices
        return self.table(flat_idx)

    @property
    def subject_tables(self):
        class SubjectTablesDict:
            def __init__(self, parent):
                self.parent = parent
            def __getitem__(self, key):
                subj_id = int(key)
                num_slices = self.parent.subject_slices[subj_id]
                return SliceEmbeddingView(self.parent.table, subj_id, self.parent.max_slices, num_slices)
            def keys(self):
                return [str(k) for k in self.parent.subject_slices.keys()]
            def __contains__(self, key):
                return str(key) in self.keys()
            def __iter__(self):
                return iter(self.keys())
        return SubjectTablesDict(self)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        old_keys = [k for k in state_dict.keys() if k.startswith(prefix + "subject_tables.")]
        if old_keys:
            with torch.no_grad():
                self.table.weight.fill_(0.0)
                for old_key in old_keys:
                    parts = old_key[len(prefix + "subject_tables."):].split(".")
                    subj_id = int(parts[0])
                    old_weight = state_dict.pop(old_key)
                    num_slices = old_weight.shape[0]
                    self.table.weight[subj_id * self.max_slices : subj_id * self.max_slices + num_slices] = old_weight
            state_dict[prefix + "table.weight"] = self.table.weight
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class MultiHeadNetwork(nn.Module):
    def __init__(self, subject_slices: dict, num_subjects, num_labels, latent_size, motion_size, hidden_size=512,
                 num_res_layers=2, linear_head=True, skip_correction=False, use_trig_encoding=True, riemannian=True):
        super(MultiHeadNetwork, self).__init__()
        self.num_labels = num_labels
        self.latent_size = latent_size
        self.motion_size = motion_size
        self.hidden_size = hidden_size
        self.linear_head = linear_head
        self.num_res_layers = num_res_layers
        self.skip_correction = skip_correction
        self.use_trig_encoding = use_trig_encoding
        self.riemannian = riemannian

        if not self.riemannian:
            #--------Latent code for baseline (no riemannian approach)---------

            # Latent code for each subject (learnable)
            self.h_init_std = 0.01 /  math.sqrt(self.latent_size)
            self.shape_code = torch.nn.Embedding(num_subjects, self.latent_size, max_norm=None).requires_grad_(True)
            torch.nn.init.normal_(self.shape_code.weight.data, 0.0, self.h_init_std)

            # Shape mean
            self.shape_mean = nn.Parameter(torch.zeros(self.latent_size), requires_grad=False)

            self.manifold = None
            self.motion_manifold = None
        else:
            #----------Latent code for each subject (learnable) - Riemannian space--------
            self.manifold = geoopt.PoincareBall(c=1.0)
            self.shape_code = RiemannianEmbedding(num_subjects, self.latent_size, self.manifold)

            # Motion manifold - Spherical space
            self.motion_manifold = geoopt.Sphere()

            # Shape mean
            self.shape_mean = geoopt.ManifoldParameter(
                torch.zeros(self.latent_size), manifold=self.manifold, requires_grad=False
            )

        # ====== Slice-motion correction ======
        self.slice_correction = SliceCorrection(subject_slices)

        # ======= Embeddings =======

        # Coordinate embedding [coordinates + time] -- or [coordinates + time + sin(time) + cos(time)] -- or position embedding
        if self.use_trig_encoding:
            self.coord_embedding = TimeTrigEncoding() # Coords + trig(time)
            trig_size = 1 # trig encoding size
        else:
            self.coord_embedding = lambda x: x  # Identity for now
            trig_size = 0 # trig encoding size

        # Latent space projection
        coordinates_size = 3  # X, Y, Z
        time_size = 1  # Time 
        input_size = coordinates_size + time_size + trig_size
        self.latent_projection = INR(input_size, self.latent_size, self.latent_size, self.latent_size, 
                                     num_hidden_layers=self.num_res_layers, use_residual=True)

        # ======= Prediction Heads =======

        # Segmentation Head
        if self.linear_head:
            self.segmentation_head = nn.Linear(self.latent_size, num_labels)
        else:
            self.segmentation_head = nn.Sequential(
                nn.Linear(self.latent_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, num_labels)
            )

        # Intensity Head
        if self.linear_head:
            self.intensity_head = nn.Linear(self.latent_size, 1)
        else:
            self.intensity_head = nn.Sequential(
                nn.Linear(self.latent_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, 1)
            )

        # ------- Motion and Displacement Field Heads -------

        # Motion Code        
        motion_input_size = self.latent_size + (2 if use_trig_encoding else 1)
        self.motion_head = nn.Sequential(
            nn.Linear(motion_input_size, motion_size),
            # nn.SiLU(),
        )

        # Displacement Field Head
        if self.linear_head:
            self.displacement_head = nn.Linear(hidden_size + motion_size, 3)
        else:
            self.displacement_head = nn.Sequential(
                nn.Linear(self.latent_size + motion_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, 3)  # Output displacement vector
            )

        # Initialize weights        
        # for m in self.modules():
        #     self.weights_init(m)
    

    def weights_init(self, m):
        """ Initialization of linear layers """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def _get_corrected_coords(self, coords, sample_idx, slice_indices, max_translation=0.2):
        # corrections: (N, 3) → angle θ, tx, ty
        corrections = self.slice_correction(sample_idx, slice_indices)  # (N, 3)
        
        theta = corrections[:, 0]                 # rotation angle
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        mat = torch.stack([cos_t, -sin_t, sin_t, cos_t], dim=-1).view(-1, 2, 2)  # (N,2,2)

        t = torch.tanh(corrections[:, 1:3]) * max_translation  # keep translations small ([-0.1,0.1])

        xy  = coords[:, 0:2].unsqueeze(-1)       # (N,2,1)
        corrected_xy = torch.bmm(mat, xy).squeeze(-1) + t

        corrected_coords = coords.clone()
        corrected_coords[:, 0:2] = corrected_xy
        return corrected_coords

    def _get_inverse_corrected_coords(self, coords, sample_idx, slice_indices, max_translation=0.2):
        # Inverse mapping: if F(x) = R * x + t, then F^(-1)(y) = R^(-1) * (y - t)
        corrections = self.slice_correction(sample_idx, slice_indices)  # (N, 3)
        
        theta = corrections[:, 0]                 # rotation angle
        cos_t, sin_t = torch.cos(-theta), torch.sin(-theta)
        mat_inv = torch.stack([cos_t, -sin_t, sin_t, cos_t], dim=-1).view(-1, 2, 2)  # (N,2,2)

        t = torch.tanh(corrections[:, 1:3]) * max_translation  # keep translations small ([-0.1,0.1])

        xy  = coords[:, 0:2].unsqueeze(-1)       # (N,2,1)
        xy_shifted = xy - t.unsqueeze(-1)
        misaligned_xy = torch.bmm(mat_inv, xy_shifted).squeeze(-1)

        misaligned_coords = coords.clone()
        misaligned_coords[:, 0:2] = misaligned_xy
        return misaligned_coords

    def decode_latent(self, corrected_coords, time, h):
        if self.use_trig_encoding:
            # Embed coordinates, time and trig(time)
            input_data = self.coord_embedding(corrected_coords, time)
        else:
            # Embed coordinates and time 
            input_data = self.coord_embedding(torch.cat([corrected_coords.float(), time.float()], dim=1))

        # Latent projection
        # print(h.shape)
        # print(input_data.shape)
        latent_t = self.latent_projection((input_data, h))

        return latent_t


    def forward(self, coords, time, sample_idx, slice_indices):
        # Correction code
        # --- Per-slice corrections ---
        if self.skip_correction:
            corrected_coords = coords
        else:
            corrected_coords = self._get_corrected_coords(coords, sample_idx, slice_indices)

        # Shape code
        h = self.shape_code(sample_idx)

        # Embed coordinates and time        
        if self.use_trig_encoding:
            input_data = self.coord_embedding(corrected_coords, time)
        else:
            input_data = self.coord_embedding(torch.cat([corrected_coords.float(), time.float()], dim=1))

        # Get the latent projection  
        latent_t = self.latent_projection((input_data, h))

        # Predict segmentation
        seg_pred = self.segmentation_head(latent_t)
        # seg_pred = nn.functional.softmax(seg_pred, dim=1)

        # Predict intensity
        intensity_pred = self.intensity_head(latent_t)

        # Apply softplus to keep values in (0,1)
        intensity_pred = F.sigmoid(intensity_pred)

        # Compute motion code
        if self.use_trig_encoding:
            sin_t = torch.sin(2*math.pi*time).float()
            cos_t = torch.cos(2*math.pi*time).float()
            motion_code = self.motion_head(torch.cat([latent_t, sin_t, cos_t], dim=1))
        else:
            motion_code = self.motion_head(torch.cat([latent_t, time.float()], dim=1))
        # Spherical constraint: project motion code onto the unit hypersphere
        if self.riemannian:
            motion_code = self.motion_manifold.projx(motion_code)

        # Compute displacement field
        motion_latent = torch.cat((latent_t, motion_code), dim=-1)
        displacement = self.displacement_head(motion_latent)
        displacement = 0.5 * torch.tanh(0.5 * displacement)  # Log-scaled tanh ~[-0.25, 0.25]
        # displacement = torch.clamp(displacement, min=-0.2, max=0.2)
        # displacement = 0.5 * torch.tanh(displacement) # Scale to [-0.5, 0.5] range / prevent large displacements

        return seg_pred, intensity_pred, displacement, h
