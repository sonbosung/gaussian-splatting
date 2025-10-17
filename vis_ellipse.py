
import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-8

# ---------- Geometry / Linear algebra ----------

def project_points(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, Xw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    Xc = R @ Xw + t
    Z = Xc[2].clamp_min(EPS)
    x_norm = Xc[:2] / Z
    x_pix = K[:2,:2] @ x_norm + K[:2,2]
    return x_pix, Xc

def project_jacobian(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, Xw: torch.Tensor) -> torch.Tensor:
    Xc = R @ Xw + t
    Z = Xc[2].clamp_min(EPS)
    fx, fy = K[0,0], K[1,1]

    dxdXc = torch.tensor([[1.0/Z, 0.0, -Xc[0]/(Z*Z)],
                          [0.0, 1.0/Z, -Xc[1]/(Z*Z)]],
                         dtype=K.dtype, device=K.device)
    A = torch.zeros((2,2), dtype=K.dtype, device=K.device)
    A[0,0] = fx
    A[1,1] = fy
    J = A @ dxdXc @ R
    return J

def compute_fundamental_from_poses(Ki: torch.Tensor, Ri: torch.Tensor, ti: torch.Tensor,
                                   Kr: torch.Tensor, Rr: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
    def skew(v: torch.Tensor) -> torch.Tensor:
        vx, vy, vz = v[0], v[1], v[2]
        return torch.tensor([[0.0, -vz,  vy],
                             [ vz,  0.0, -vx],
                             [-vy,  vx,  0.0]], dtype=v.dtype, device=v.device)

    R_rel = Ri @ Rr.T
    t_rel = ti - R_rel @ tr
    E = skew(t_rel) @ R_rel
    Ki_inv = torch.inverse(Ki)
    Kr_inv = torch.inverse(Kr)
    Fmat = Ki_inv.T @ E @ Kr_inv
    return Fmat

# ---------- SPD utilities ----------

def spd_logm_2x2(S: torch.Tensor) -> torch.Tensor:
    S = 0.5*(S + S.transpose(-1,-2))
    evals, evecs = torch.linalg.eigh(S)
    evals = evals.clamp_min(EPS)
    logD = torch.diag_embed(torch.log(evals))
    return evecs @ logD @ evecs.transpose(-1,-2)

# ---------- Gaussian window moments (sigma-differentiable) ----------

def gaussian_window_moments(image, center_xy, sigma, patch_mult=3.0, contrast='plain'):
    assert image.ndim in (2,4)
    if image.ndim == 2:
        img = image[None, None]
    else:
        img = image

    H, W = img.shape[-2:]
    rad = torch.clamp((patch_mult * sigma).ceil().long(), min=3)
    xs = torch.arange(-rad.item(), rad.item()+1, device=img.device, dtype=img.dtype)
    ys = torch.arange(-rad.item(), rad.item()+1, device=img.device, dtype=img.dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')

    cx, cy = center_xy[0], center_xy[1]
    X = cx + grid_x;  Y = cy + grid_y
    gx = (X / (W - 1) * 2 - 1).clamp(-1, 1)
    gy = (Y / (H - 1) * 2 - 1).clamp(-1, 1)
    grid = torch.stack([gx, gy], dim=-1)[None]
    patch = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)[0,0]  # (M,M)

    inv_2s2 = 1.0 / (2.0 * sigma * sigma + 1e-8)
    w = torch.exp(-(grid_x*grid_x + grid_y*grid_y) * inv_2s2)  # Gaussian window

    # 내용 기반 가중치: 대비가 있는 곳을 더 크게
    if contrast == 'absmean':
        c = (patch - patch.mean()).abs()
    elif contrast == 'range':
        c = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
    else:  # 'plain' (그냥 intensity)
        c = patch.clamp_min(0)

    Wtot = (w * (c + 1e-8))        # 내용 가중치 곱
    w_sum = Wtot.sum() + 1e-8

    mx = (Wtot * X).sum() / w_sum
    my = (Wtot * Y).sum() / w_sum
    dx = X - mx;  dy = Y - my

    cxx = (Wtot * dx * dx).sum() / w_sum
    cyy = (Wtot * dy * dy).sum() / w_sum
    cxy = (Wtot * dx * dy).sum() / w_sum

    S = torch.stack([torch.stack([cxx, cxy]), torch.stack([cxy, cyy])])
    return S + torch.eye(2, device=S.device, dtype=S.dtype) * 1e-9

def structure_tensor_cov_swapped(image: torch.Tensor,
                                 center_xy: torch.Tensor,
                                 sigma: torch.Tensor,
                                 patch_mult: float = 3.0,
                                 eps: float = 1e-6,
                                 normalize_to: str = 'none') -> torch.Tensor:
    """
    Create a 2x2 covariance matrix (ellipse) from the local structure tensor,
    with eigenvalues swapped so strong-gradient directions appear NARROW.

    Args:
        image: (H,W) or (1,1,H,W) grayscale tensor in [0,1]
        center_xy: (2,) [u,v] pixel center
        sigma: scalar tensor (>0), also used as the Gaussian window sigma (in pixels)
        patch_mult: radius = ceil(patch_mult * sigma)
        eps: small Tikhonov for numerical stability
        normalize_to: 'sigma' or 'none'
            - 'sigma': scale Sigma so that trace(Sigma) ≈ 2*sigma^2
            - 'none' : leave raw scale from the swapped structure tensor

    Returns:
        Sigma (2,2) SPD
    """
    assert image.ndim in (2,4)
    if image.ndim == 2:
        img = image[None, None]
    else:
        img = image

    H, W = img.shape[-2:]
    # window/grid
    rad = torch.clamp((patch_mult * sigma).ceil().long(), min=3)
    xs = torch.arange(-rad.item(), rad.item()+1, device=img.device, dtype=img.dtype)
    ys = torch.arange(-rad.item(), rad.item()+1, device=img.device, dtype=img.dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')

    cx, cy = center_xy[0], center_xy[1]
    X = cx + grid_x
    Y = cy + grid_y
    gx = (X / (W - 1) * 2 - 1).clamp(-1, 1)
    gy = (Y / (H - 1) * 2 - 1).clamp(-1, 1)
    grid = torch.stack([gx, gy], dim=-1)[None]
    patch = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)[0,0]  # (M,M)

    # Sobel gradients
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=patch.dtype, device=patch.device) / 8.0
    ky = kx.t()
    Gx = F.conv2d(patch[None,None], kx[None,None], padding=1)[0,0]
    Gy = F.conv2d(patch[None,None], ky[None,None], padding=1)[0,0]

    # Gaussian window weights
    inv_2s2 = 1.0 / (2.0 * sigma * sigma + 1e-8)
    w = torch.exp(-(grid_x*grid_x + grid_y*grid_y) * inv_2s2)
    w_sum = w.sum() + 1e-8

    # Structure tensor components (window-weighted average)
    J11 = (w * (Gx*Gx)).sum() / w_sum
    J22 = (w * (Gy*Gy)).sum() / w_sum
    J12 = (w * (Gx*Gy)).sum() / w_sum
    J = torch.stack([torch.stack([J11, J12]), torch.stack([J12, J22])])

    # Stabilize
    J = 0.5*(J + J.T) + torch.eye(2, dtype=J.dtype, device=J.device)*eps

    # Eigen-decompose and swap eigenvalues
    lam, V = torch.linalg.eigh(J)   # ascending: lam[0] <= lam[1]
    lam_swapped = torch.flip(lam, dims=[0])  # [lam1, lam0]
    Sigma = V @ torch.diag(lam_swapped) @ V.T

    # Optional scale normalization (to roughly match isotropic sigma^2 scale)
    if normalize_to == 'sigma':
        target_trace = 2.0 * (sigma * sigma)
        s = (target_trace / (torch.trace(Sigma) + 1e-12))
        Sigma = s * Sigma

    # Ensure SPD
    Sigma = 0.5*(Sigma + Sigma.T) + torch.eye(2, dtype=Sigma.dtype, device=Sigma.device)*1e-9
    return Sigma

# ---------- Ellipse helpers ----------

def ellipse_params_from_cov(S: torch.Tensor, std_scale: float = 2.0) -> Tuple[float, float, float]:
    S = 0.5*(S + S.T)
    evals, evecs = torch.linalg.eigh(S)  # ascending
    l1 = evals[1].clamp_min(EPS)
    l2 = evals[0].clamp_min(EPS)
    v1 = evecs[:,1]
    theta = math.atan2(float(v1[1]), float(v1[0]))
    width = float(std_scale * math.sqrt(l1))
    height = float(std_scale * math.sqrt(l2))
    return width, height, theta

def draw_ellipse(ax, center_xy, cov_2x2, std_scale: float = 2.0, label: Optional[str] = None):
    import matplotlib.patches as patches
    w, h, theta = ellipse_params_from_cov(cov_2x2, std_scale=std_scale)
    e = patches.Ellipse((float(center_xy[0]), float(center_xy[1])),
                        2*w, 2*h,
                        angle=np.degrees(theta), fill=False)
    ax.add_patch(e)
    if label is not None:
        e.set_label(label)
    ax.plot([float(center_xy[0])], [float(center_xy[1])], marker='x')

def draw_sigma_window(ax, center_xy, sigma: float, k: float = 3.0):
    import matplotlib.patches as patches
    rad = float(k * sigma)
    circ = patches.Circle((float(center_xy[0]), float(center_xy[1])), radius=rad, fill=False, linestyle='--')
    ax.add_patch(circ)

# ---------- Visualization 1: per-view predicted vs measured ellipses ----------

def visualize_ellipse_projection(
    Ks: List[torch.Tensor],
    Rs: List[torch.Tensor],
    ts: List[torch.Tensor],
    images: List[torch.Tensor],
    mu_3d: torch.Tensor,
    Sigma3D: torch.Tensor,
    sigma_ws: List[float],
    std_scale: float = 2.0,
    patch_mult: float = 3.0,
    figsize_per_view: Tuple[int,int] = (5,5),
    show_sigma_window: bool = True,
    window_k: float = 3.0,
    zoom_to_window: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, List]:
    n = len(Ks)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_view[0]*cols, figsize_per_view[1]*rows))
    if n == 1:
        axes = np.array([[axes]])
    axes = np.array(axes).reshape(rows, cols)

    diag = {'centers': [], 'S_pred': [], 'S_meas': []}

    for i in range(n):
        r = i // cols
        c = i % cols
        ax = axes[r, c]

        Ki, Ri, ti = Ks[i], Rs[i], ts[i]
        img = images[i]
        if img.ndim == 2:
            imshow = img.detach().cpu().numpy()
        else:
            imshow = img[0,0].detach().cpu().numpy()

        ax.imshow(imshow, origin='upper')

        xi_pix, _ = project_points(Ki, Ri, ti, mu_3d)
        J = project_jacobian(Ki, Ri, ti, mu_3d)
        S_pred = J @ Sigma3D @ J.T

        sigma_i = float(sigma_ws[i])
        S_meas = structure_tensor_cov_swapped(img, xi_pix, torch.tensor(sigma_i, dtype=img.dtype, device=img.device), patch_mult=patch_mult)

        draw_ellipse(ax, xi_pix, S_pred, std_scale=std_scale, label='pred JΣJᵀ')
        draw_ellipse(ax, xi_pix, S_meas, std_scale=std_scale, label='meas moments')

        if show_sigma_window:
            draw_sigma_window(ax, xi_pix, sigma=sigma_i, k=window_k)

        ax.set_title(f"View {i}")
        H, W = imshow.shape
        if zoom_to_window:
            rad = window_k * sigma_i
            margin = max(8.0, 0.2 * rad)
            ax.set_xlim([float(xi_pix[0]-rad-margin), float(xi_pix[0]+rad+margin)])
            ax.set_ylim([float(xi_pix[1]+rad+margin), float(xi_pix[1]-rad-margin)])
        else:
            ax.set_xlim([0, W-1])
            ax.set_ylim([H-1, 0])
        ax.legend(loc='upper right')

        diag['centers'].append(xi_pix.detach().cpu())
        diag['S_pred'].append(S_pred.detach().cpu())
        diag['S_meas'].append(S_meas.detach().cpu())

    for j in range(n, rows*cols):
        r = j // cols; c = j % cols
        axes[r, c].axis('off')

    fig.suptitle("Predicted vs Measured Ellipses per View (zoomed to σ-window)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return diag

# ---------- Visualization 2: epipolar normal scale comparison ----------

def visualize_epipolar_normal_scales(
    Ks: List[torch.Tensor],
    Rs: List[torch.Tensor],
    ts: List[torch.Tensor],
    images: List[torch.Tensor],
    mu_3d: torch.Tensor,
    Sigma3D: torch.Tensor,
    sigma_ws: List[float],
    ref_index: int = 0,
    gammas: Optional[List[float]] = None,
    std_scale: float = 2.0,
    patch_mult: float = 3.0,
    figsize_per_view: Tuple[int,int] = (5,5),
    show_sigma_window: bool = True,
    window_k: float = 3.0,
    zoom_to_window: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, List]:
    n = len(Ks)
    num_targets = max(0, n-1)
    cols = min(3, max(1, num_targets))
    rows = math.ceil(max(1, num_targets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_view[0]*cols, figsize_per_view[1]*rows))
    if num_targets == 1:
        axes = np.array([[axes]])
    axes = np.array(axes).reshape(rows, cols) if num_targets > 0 else np.array([])

    Kr, Rr, tr = Ks[ref_index], Rs[ref_index], ts[ref_index]
    xr_pix, _ = project_points(Kr, Rr, tr, mu_3d)
    xr_h = torch.tensor([xr_pix[0], xr_pix[1], 1.0], dtype=Kr.dtype, device=Kr.device)

    diag = {'v_pred': [], 'v_meas': [], 'v_meas_gamma': [], 'n_vec': []}

    subplot_idx = 0
    for i in range(n):
        if i == ref_index:
            continue
        Ki, Ri, ti = Ks[i], Rs[i], ts[i]
        img = images[i]
        if img.ndim == 2:
            imshow = img.detach().cpu().numpy()
        else:
            imshow = img[0,0].detach().cpu().numpy()

        r = subplot_idx // cols
        c = subplot_idx % cols
        ax = axes[r, c]
        ax.imshow(imshow, origin='upper')

        Fmat = compute_fundamental_from_poses(Ki, Ri, ti, Kr, Rr, tr)
        ell = Fmat @ xr_h  # (a,b,c)
        a, b, c0 = float(ell[0]), float(ell[1]), float(ell[2])

        H, W = imshow.shape
        pts = []
        if abs(b) > 1e-9:
            y0 = -(a*0 + c0)/b
            y1 = -(a*(W-1) + c0)/b
            pts.extend([(0, y0), (W-1, y1)])
        if abs(a) > 1e-9:
            x0 = -(b*0 + c0)/a
            x1 = -(b*(H-1) + c0)/a
            pts.extend([(x0, 0), (x1, H-1)])
        pts = [(x,y) for (x,y) in pts if -W*0.5 <= x <= W*1.5 and -H*0.5 <= y <= H*1.5]
        if len(pts) >= 2:
            (xA,yA),(xB,yB) = pts[0], pts[1]
            ax.plot([xA,xB],[yA,yB])

        n_vec = np.array([a,b], dtype=np.float64)
        n_norm = np.linalg.norm(n_vec) + 1e-12
        n_unit = n_vec / n_norm

        xi_pix, _ = project_points(Ki, Ri, ti, mu_3d)

        J = project_jacobian(Ki, Ri, ti, mu_3d)
        S_pred = J @ Sigma3D @ J.T
        sigma_i = float(sigma_ws[i])
        S_meas = structure_tensor_cov_swapped(img, xi_pix, torch.tensor(sigma_i, dtype=img.dtype, device=img.device), patch_mult=patch_mult)

        n_t = torch.tensor(n_unit, dtype=img.dtype, device=img.device)
        v_pred = (n_t[None] @ S_pred @ n_t[:,None]).squeeze()
        v_meas = (n_t[None] @ S_meas @ n_t[:,None]).squeeze()

        gamma_i = float(gammas[i]) if (gammas is not None) else 1.0
        v_meas_gamma = v_meas * gamma_i

        # draw
        draw_ellipse(ax, xi_pix, S_pred, std_scale=std_scale, label="pred")
        draw_ellipse(ax, xi_pix, S_meas, std_scale=std_scale, label="meas")
        if show_sigma_window:
            draw_sigma_window(ax, xi_pix, sigma=sigma_i, k=window_k)
        ax.arrow(float(xi_pix[0]), float(xi_pix[1]),
                 float(n_unit[0])*20.0, float(n_unit[1])*20.0,
                 head_width=3.0, length_includes_head=True)

        ax.set_title(f"View {i}")
        if zoom_to_window:
            rad = window_k * sigma_i
            margin = max(8.0, 0.2 * rad)
            ax.set_xlim([float(xi_pix[0]-rad-margin), float(xi_pix[0]+rad+margin)])
            ax.set_ylim([float(xi_pix[1]+rad+margin), float(xi_pix[1]-rad-margin)])
        else:
            ax.set_xlim([0, W-1])
            ax.set_ylim([H-1, 0])
        ax.legend(loc='upper right')

        diag['v_pred'].append(float(v_pred.detach().cpu()))
        diag['v_meas'].append(float(v_meas.detach().cpu()))
        diag['v_meas_gamma'].append(float(v_meas_gamma.detach().cpu()))
        diag['n_vec'].append(n_unit.tolist())

        subplot_idx += 1

    fig.suptitle("Epipolar Normal-Direction Scale Comparison (zoomed to σ-window)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return diag
