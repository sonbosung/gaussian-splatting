# ellipse_batch_estimator_point_epoch.py
# -----------------------------------------------------------------------------
# Point-batched estimator (epoch-based) for per-point 3D covariance from
# multi-view observations with per-observation learnable sigma.
#
# - Epoch-based: shuffle points each epoch, take 'batch_points' per step
# - For selected points, gather all/capped observations -> variable obs batch
# - Structure-tensor measurement (view-grouped & chunked) + eigenvalue swap
# - Pairwise (relative-only) epipolar-normal-scale loss
# - Per-view scale gamma, per-observation sigma (softplus)
# - AMP, grad-clip, eigenvector-detach, clamp for stability
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# =========================
# Helpers
# =========================

def cholesky_param_to_spd(L_vec: torch.Tensor) -> torch.Tensor:
    l00, l10, l11, l20, l21, l22 = torch.unbind(L_vec, dim=-1)
    L = torch.zeros((*L_vec.shape[:-1], 3, 3), dtype=L_vec.dtype, device=L_vec.device)
    L[..., 0, 0] = torch.exp(l00)
    L[..., 1, 0] = l10
    L[..., 1, 1] = torch.exp(l11)
    L[..., 2, 0] = l20
    L[..., 2, 1] = l21
    L[..., 2, 2] = torch.exp(l22)
    S = L @ L.transpose(-1, -2)
    return S + torch.eye(3, device=L.device, dtype=L.dtype) * 1e-12


def projection_jacobian(Kv: torch.Tensor, Rv: torch.Tensor, tv: torch.Tensor, Xw: torch.Tensor) -> torch.Tensor:
    Xc = (Rv @ Xw.T).T + tv  # (B,3)
    Z = Xc[:, 2].clamp_min(1e-6)
    fx, fy = Kv[0, 0], Kv[1, 1]
    dxdXc = torch.stack([
        torch.stack([1.0 / Z, torch.zeros_like(Z), -Xc[:, 0] / (Z * Z)], dim=-1),
        torch.stack([torch.zeros_like(Z), 1.0 / Z, -Xc[:, 1] / (Z * Z)], dim=-1),
    ], dim=1)  # (B,2,3)
    A = torch.zeros((2, 2), dtype=Xw.dtype, device=Xw.device)
    A[0, 0] = fx; A[1, 1] = fy
    J = torch.einsum('ij,bjk->bik', A, dxdXc) @ Rv  # (B,2,3)
    return J


# =========================
# Structure-tensor measurement (streaming)
# =========================

def measure_covariances_batched_varsigma_streaming(
    images: torch.Tensor,            # (V,1,H,W)
    obs_view_idx: torch.Tensor,      # (M,)
    centers_xy: torch.Tensor,        # (M,2)
    sigma_obs: torch.Tensor,         # (M,)
    *,
    patch_mult: float = 1.0,
    normalize_to: str = 'none',
    max_chunk_obs: int = 4096,
    r_cap: int = 31,
    use_amp: bool = True,
    detach_evecs: bool = True
) -> torch.Tensor:
    device, dtype = images.device, images.dtype
    V, C, H, W = images.shape
    assert C == 1, "images must be (V,1,H,W)."

    # Sobel gradients
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    if use_amp:
        with torch.cuda.amp.autocast():
            Ix_all = F.conv2d(images, kx, padding=1)
            Iy_all = F.conv2d(images, ky, padding=1)
    else:
        Ix_all = F.conv2d(images, kx, padding=1)
        Iy_all = F.conv2d(images, ky, padding=1)

    sigma_obs = torch.nan_to_num(sigma_obs, nan=1.0, posinf=50.0, neginf=1.0).clamp(min=1e-3)
    r = int(torch.clamp((patch_mult * sigma_obs.max()).ceil(), min=3, max=r_cap).item())
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    ys = torch.arange(-r, r + 1, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')

    M = centers_xy.shape[0]
    Ixx = torch.empty(M, device=device, dtype=dtype)
    Iyy = torch.empty(M, device=device, dtype=dtype)
    Ixy = torch.empty(M, device=device, dtype=dtype)

    for v in torch.unique(obs_view_idx).tolist():
        mask_v = (obs_view_idx == v)
        idx_v = torch.nonzero(mask_v, as_tuple=False).squeeze(1)
        if idx_v.numel() == 0:
            continue

        Ix_v = Ix_all[v:v + 1]
        Iy_v = Iy_all[v:v + 1]

        for cstart in range(0, idx_v.numel(), max_chunk_obs):
            cend = min(cstart + max_chunk_obs, idx_v.numel())
            inds = idx_v[cstart:cend]
            m = int(inds.numel())
            cx = centers_xy[inds, 0:1]
            cy = centers_xy[inds, 1:2]
            sig = sigma_obs[inds].view(m, 1, 1)

            X = cx.view(m, 1, 1) + grid_x
            Y = cy.view(m, 1, 1) + grid_y
            gx = (X / (W - 1) * 2 - 1).clamp(-1, 1)
            gy = (Y / (H - 1) * 2 - 1).clamp(-1, 1)
            grid = torch.stack([gx, gy], dim=-1)

            Ix_patch = F.grid_sample(Ix_v.expand(m, -1, -1, -1), grid, mode='bilinear',
                                     padding_mode='border', align_corners=True)[:, 0]
            Iy_patch = F.grid_sample(Iy_v.expand(m, -1, -1, -1), grid, mode='bilinear',
                                     padding_mode='border', align_corners=True)[:, 0]

            den = (2.0 * sig**2).clamp(min=1e-5)
            w = torch.exp(-(grid_x**2 + grid_y**2)[None, :, :] / den)
            wsum = w.sum(dim=(1, 2)).clamp(min=1e-5)

            Ixx[inds] = (w * (Ix_patch ** 2)).sum(dim=(1, 2)) / wsum
            Iyy[inds] = (w * (Iy_patch ** 2)).sum(dim=(1, 2)) / wsum
            Ixy[inds] = (w * (Ix_patch * Iy_patch)).sum(dim=(1, 2)) / wsum

    M11, M22, M12 = Ixx, Iyy, Ixy
    Mmat = torch.stack([torch.stack([M11, M12], dim=-1),
                        torch.stack([M12, M22], dim=-1)], dim=-2)
    Mmat = 0.5 * (Mmat + Mmat.transpose(-1, -2)) + torch.eye(2, device=device, dtype=dtype) * 1e-12

    evals, evecs = torch.linalg.eigh(Mmat)
    evals = torch.clamp(evals, min=1e-12)
    if detach_evecs:
        evecs = evecs.detach()
    evals_swapped = torch.flip(evals, dims=[-1])
    S = (evecs @ torch.diag_embed(evals_swapped) @ evecs.transpose(-1, -2))

    side_iso = (2.0 * sigma_obs + 1.0)
    A = torch.zeros(S.shape[0], 2, 2, device=device, dtype=dtype)
    A[:, 0, 0] = side_iso
    A[:, 1, 1] = side_iso
    S = A @ S @ A.transpose(-1, -2)

    if normalize_to == 'sigma':
        target = 2.0 * (sigma_obs ** 2)
        tr = (S[..., 0, 0] + S[..., 1, 1]).clamp_min(1e-9)
        S = S * (target / tr).view(-1, 1, 1)

    S = 0.5 * (S + S.transpose(-1, -2)) + torch.eye(2, device=device, dtype=dtype) * 1e-9
    return S


# =========================
# Losses
# =========================

def loss_shape_trace_norm(Sp: torch.Tensor, Sm: torch.Tensor) -> torch.Tensor:
    tp = (Sp[..., 0, 0] + Sp[..., 1, 1]).clamp_min(1e-9).unsqueeze(-1).unsqueeze(-1)
    tm = (Sm[..., 0, 0] + Sm[..., 1, 1]).clamp_min(1e-9).unsqueeze(-1).unsqueeze(-1)
    return ((Sp / tp - Sm / tm) ** 2).mean()


def loss_scale_logtrace(Sp: torch.Tensor, Sm: torch.Tensor) -> torch.Tensor:
    tp = (Sp[..., 0, 0] + Sp[..., 1, 1]).clamp_min(1e-9)
    tm = (Sm[..., 0, 0] + Sm[..., 1, 1]).clamp_min(1e-9)
    return ((torch.log(tp) - torch.log(tm)) ** 2).mean()


def epipolar_normal_scale_loss_pairwise(
    K: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
    obs_pidx: torch.Tensor, obs_vidx: torch.Tensor,
    centers_xy: torch.Tensor, S_pred: torch.Tensor, S_meas: torch.Tensor,
    *, max_pairs_per_point: int = 2, bidirectional: bool = True, weight: float = 1.0
) -> torch.Tensor:
    device = K.device; dtype = K.dtype
    Mb = obs_pidx.shape[0]
    if Mb == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    Kinv = torch.linalg.inv(K)

    from collections import defaultdict
    by_point: Dict[int, List[int]] = defaultdict(list)
    for k in range(Mb):
        by_point[int(obs_pidx[k].item())].append(k)

    errs = []

    for _, idxs in by_point.items():
        if len(idxs) < 2:
            continue
        sel = idxs[:max_pairs_per_point + 1] if len(idxs) > max_pairs_per_point + 1 else idxs
        m = len(sel)
        if m < 2:
            continue

        for a in range(m):
            i = sel[a]
            vi = int(obs_vidx[i].item())
            Ki_inv = Kinv[vi]
            Ri, ti = R[vi], t[vi]
            xi = torch.stack([centers_xy[i, 0], centers_xy[i, 1],
                              torch.tensor(1.0, device=device, dtype=dtype)])

            for b in range(a + 1, m):
                j = sel[b]
                vj = int(obs_vidx[j].item())
                Kj_inv = Kinv[vj]
                Rj, tj = R[vj], t[vj]

                R_ji = Rj @ Ri.transpose(-1, -2)
                t_ji = tj - R_ji @ ti
                tx = torch.tensor([[0, -t_ji[2],  t_ji[1]],
                                   [t_ji[2], 0, -t_ji[0]],
                                   [-t_ji[1], t_ji[0], 0]], device=device, dtype=dtype)
                F_ij = Kj_inv.transpose(-1, -2) @ tx @ R_ji @ Ki_inv

                l_j = F_ij @ xi
                dj = l_j[:2]; dj = dj / (dj.norm() + 1e-12)
                vv_pred_j = dj @ S_pred[j] @ dj
                vv_meas_j = dj @ S_meas[j] @ dj
                errs.append((torch.log(vv_pred_j.clamp_min(1e-12)) - torch.log(vv_meas_j.clamp_min(1e-12))) ** 2)

                if bidirectional:
                    xj = torch.stack([centers_xy[j, 0], centers_xy[j, 1],
                                      torch.tensor(1.0, device=device, dtype=dtype)])
                    R_ij = Ri @ Rj.transpose(-1, -2)
                    t_ij = ti - R_ij @ tj
                    tx2 = torch.tensor([[0, -t_ij[2],  t_ij[1]],
                                        [t_ij[2], 0, -t_ij[0]],
                                        [-t_ij[1], t_ij[0], 0]], device=device, dtype=dtype)
                    F_ji = Ki_inv.transpose(-1, -2) @ tx2 @ R_ij @ Kinv[vj]
                    l_i = F_ji @ xj
                    di = l_i[:2]; di = di / (di.norm() + 1e-12)
                    vv_pred_i = di @ S_pred[i] @ di
                    vv_meas_i = di @ S_meas[i] @ di
                    errs.append((torch.log(vv_pred_i.clamp_min(1e-12)) - torch.log(vv_meas_i.clamp_min(1e-12))) ** 2)

    if len(errs) == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return weight * torch.stack(errs).mean()


# =========================
# Config
# =========================

@dataclass
class EstimatorConfig:
    # measurement
    patch_mult: float = 1.0
    normalize_to: str = 'none'
    r_cap: int = 31
    max_chunk_obs: int = 4096
    detach_evecs: bool = True

    # optimization (epoch-based)
    epochs: int = 10
    batch_points: int = 1024           # number of points per step
    max_obs_per_point: int = 8         # cap per-point observations in a step (for heavy points)
    lr: float = 1e-3
    lr_sigma: float = 1e-4
    use_amp: bool = True
    max_grad_norm: float = 1.0
    log_interval: int = 50

    # sigma param
    learn_sigma_obs: bool = True
    init_sigma: float = 4.0
    sigma_max: float = 50.0
    softplus_beta: float = 0.5

    # losses
    w_scale: float = 0.3
    lambda_epi: float = 0.1
    epi_max_pairs_per_point: int = 2
    epi_bidirectional: bool = True


# =========================
# Estimator
# =========================

class EllipsoidBatchEstimator(nn.Module):
    def __init__(
        self,
        X: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
        obs_pidx_all: torch.Tensor, obs_vidx_all: torch.Tensor,
        cfg: EstimatorConfig
    ):
        super().__init__()
        self.register_buffer('X', X)
        self.register_buffer('K', K); self.register_buffer('R', R); self.register_buffer('t', t)
        self.register_buffer('obs_pidx_all', obs_pidx_all)
        self.register_buffer('obs_vidx_all', obs_vidx_all)
        self.cfg = cfg

        N = X.shape[0]; M = obs_pidx_all.shape[0]; V = K.shape[0]

        self.L_params = nn.Parameter(torch.zeros(N, 6, dtype=X.dtype, device=X.device))
        with torch.no_grad():
            self.L_params[:, :3] = -1.0  # modest exp init

        init_sigma = torch.full((M,), cfg.init_sigma, dtype=X.dtype, device=X.device).clamp(min=1e-3)
        init_alpha = torch.log(torch.expm1(init_sigma))  # inverse softplus
        self.alpha_per_obs = nn.Parameter(init_alpha, requires_grad=cfg.learn_sigma_obs)

        self.gamma_view = nn.Parameter(torch.ones(V, dtype=X.dtype, device=X.device))

        # point -> observation indices (list of tensors on device)
        self.point2obs: List[torch.Tensor] = [torch.empty(0, dtype=torch.long, device=X.device) for _ in range(N)]
        for j in range(M):
            p = int(obs_pidx_all[j].item())
            self.point2obs[p] = torch.cat([self.point2obs[p], torch.tensor([j], device=X.device, dtype=torch.long)])

    def sigma_from_obs_idx(self, obs_idx_global: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(self.alpha_per_obs[obs_idx_global], beta=self.cfg.softplus_beta)
        return sigma.clamp(min=1e-3, max=self.cfg.sigma_max)

    def project_centers(self, pidx_b: torch.Tensor, vidx_b: torch.Tensor) -> torch.Tensor:
        K, R, t = self.K, self.R, self.t
        X_sel = self.X[pidx_b]
        centers = []
        for i in range(vidx_b.shape[0]):
            v = int(vidx_b[i].item())
            Kv, Rv, tv = K[v], R[v], t[v]
            Xc = (Rv @ X_sel[i]) + tv
            z = Xc[2].clamp_min(1e-6)
            xi = torch.stack([Kv[0, 0] * (Xc[0] / z) + Kv[0, 2],
                              Kv[1, 1] * (Xc[1] / z) + Kv[1, 2]])
            centers.append(xi)
        return torch.stack(centers, dim=0)

    def forward_obs_indices(self, images: torch.Tensor, obs_idx_global: torch.Tensor) -> Dict[str, Any]:
        pidx_b = self.obs_pidx_all[obs_idx_global]
        vidx_b = self.obs_vidx_all[obs_idx_global]
        centers_xy = self.project_centers(pidx_b, vidx_b)

        sigma_b = self.sigma_from_obs_idx(obs_idx_global)
        S_meas_raw = measure_covariances_batched_varsigma_streaming(
            images, vidx_b, centers_xy, sigma_b,
            patch_mult=self.cfg.patch_mult,
            normalize_to=self.cfg.normalize_to,
            max_chunk_obs=self.cfg.max_chunk_obs,
            r_cap=self.cfg.r_cap,
            use_amp=self.cfg.use_amp,
            detach_evecs=self.cfg.detach_evecs
        )

        gamma = self.gamma_view[vidx_b].view(-1, 1, 1)
        S_meas = gamma * S_meas_raw

        Sigma3 = cholesky_param_to_spd(self.L_params[pidx_b])
        # build J per view
        J_full = torch.zeros((vidx_b.shape[0], 2, 3), dtype=Sigma3.dtype, device=Sigma3.device)
        for v in torch.unique(vidx_b):
            mask = (vidx_b == v)
            Jv = projection_jacobian(self.K[int(v)], self.R[int(v)], self.t[int(v)], self.X[pidx_b[mask]])
            J_full[mask] = Jv
        S_pred = J_full @ Sigma3 @ J_full.transpose(-1, -2)
        S_pred = 0.5 * (S_pred + S_pred.transpose(-1, -2)) + torch.eye(2, device=S_pred.device, dtype=S_pred.dtype) * 1e-12

        return dict(S_meas=S_meas, S_pred=S_pred, pidx_b=pidx_b, vidx_b=vidx_b,
                    centers=centers_xy, obs_idx=obs_idx_global)


# =========================
# Training (epoch-based, point-batched)
# =========================

def train_ellipsoids_point_epoch(
    images: torch.Tensor,             # (V,1,H,W) or (V,H,W) grayscale
    K: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
    X: torch.Tensor,                  # (N,3)
    obs_pidx: torch.Tensor,           # (M,)
    obs_vidx: torch.Tensor,           # (M,)
    cfg: EstimatorConfig = EstimatorConfig(),
) -> Tuple[EllipsoidBatchEstimator, Dict[str, Any]]:
    device = images.device; dtype = images.dtype
    if images.ndim == 3:
        images = images.unsqueeze(1)
    assert images.ndim == 4 and images.shape[1] == 1

    model = EllipsoidBatchEstimator(X, K, R, t, obs_pidx, obs_vidx, cfg).to(device=device, dtype=dtype)

    opt = torch.optim.AdamW(
        [
            {"params": [model.L_params], "lr": cfg.lr},
            {"params": [model.alpha_per_obs], "lr": cfg.lr_sigma},
            {"params": [model.gamma_view], "lr": cfg.lr_sigma},
        ],
        eps=1e-8, amsgrad=True
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    N = X.shape[0]
    history = {"loss_total": [], "loss_shape": [], "loss_scale": []}

    for ep in range(cfg.epochs):
        # shuffle points each epoch
        order = torch.randperm(N, device=device)
        # iterate by point mini-batch
        for start in range(0, N, cfg.batch_points):
            end = min(start + cfg.batch_points, N)
            pts = order[start:end]

            # gather observations for selected points (cap per point)
            obs_list = []
            for p in pts.tolist():
                idxs = model.point2obs[p]
                if idxs.numel() == 0:
                    continue
                if idxs.numel() > cfg.max_obs_per_point:
                    perm = torch.randperm(idxs.numel(), device=device)
                    take = idxs[perm[:cfg.max_obs_per_point]]
                else:
                    take = idxs
                obs_list.append(take)
            if len(obs_list) == 0:
                continue
            obs_idx = torch.cat(obs_list, dim=0)  # variable-sized observation batch

            # forward + losses
            model.train()
            opt.zero_grad(set_to_none=True)
            if cfg.use_amp:
                with torch.cuda.amp.autocast():
                    out = model.forward_obs_indices(images, obs_idx)
                    l_shape = loss_shape_trace_norm(out["S_pred"], out["S_meas"])
                    l_scale = loss_scale_logtrace(out["S_pred"], out["S_meas"])
                    loss = l_shape + cfg.w_scale * l_scale
                    if cfg.lambda_epi > 0:
                        epi = epipolar_normal_scale_loss_pairwise(
                            K, R, t,
                            out["pidx_b"], out["vidx_b"], out["centers"],
                            out["S_pred"], out["S_meas"],
                            max_pairs_per_point=cfg.epi_max_pairs_per_point,
                            bidirectional=cfg.epi_bidirectional,
                            weight=cfg.lambda_epi
                        )
                        loss = loss + epi
            else:
                out = model.forward_obs_indices(images, obs_idx)
                l_shape = loss_shape_trace_norm(out["S_pred"], out["S_meas"])
                l_scale = loss_scale_logtrace(out["S_pred"], out["S_meas"])
                loss = l_shape + cfg.w_scale * l_scale
                if cfg.lambda_epi > 0:
                    epi = epipolar_normal_scale_loss_pairwise(
                        K, R, t,
                        out["pidx_b"], out["vidx_b"], out["centers"],
                        out["S_pred"], out["S_meas"],
                        max_pairs_per_point=cfg.epi_max_pairs_per_point,
                        bidirectional=cfg.epi_bidirectional,
                        weight=cfg.lambda_epi
                    )
                    loss = loss + epi

            history["loss_total"].append(float(loss.detach().item()))
            history["loss_shape"].append(float(l_shape.detach().item()))
            history["loss_scale"].append(float(l_scale.detach().item()))

            # backward
            if cfg.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

        # epoch summary
        if (ep + 1) % 1 == 0:
            with torch.no_grad():
                msg = (f"[epoch {ep+1}/{cfg.epochs}] "
                       f"loss={history['loss_total'][-1]:.6f} "
                       f"(shape={history['loss_shape'][-1]:.4f}, scale={history['loss_scale'][-1]:.4f})")
                print(msg)

    return model, history