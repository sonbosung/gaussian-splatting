import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

from vis_ellipse import (
    structure_tensor_cov_swapped,
    project_points,
    compute_fundamental_from_poses,
)

EPS = 1e-8

def _camera_center(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return -R.T @ t

def _plane_normal_baseline(C1: torch.Tensor, C2: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    b = C2 - C1
    bb = (b @ b).clamp_min(EPS)
    tstar = ((X - C1) @ b) / bb
    P = C1 + tstar * b
    n = P - X
    return n / (torch.norm(n) + EPS)

def _plane_basis(n: torch.Tensor):
    a = torch.tensor([1.0,0.0,0.0], dtype=n.dtype, device=n.device)
    if torch.abs(torch.dot(a,n)) > 0.9: a = torch.tensor([0.0,1.0,0.0], dtype=n.dtype, device=n.device)
    e1 = a - torch.dot(a,n)*n; e1 = e1/(torch.norm(e1)+EPS)
    e2 = torch.cross(n,e1);    e2 = e2/(torch.norm(e2)+EPS)
    return e1,e2

def _J_proj(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, Xw: torch.Tensor) -> torch.Tensor:
    Xc = R @ Xw + t
    Z = Xc[2].clamp_min(EPS)
    fx, fy = K[0,0], K[1,1]
    dxdXc = torch.tensor([[1.0/Z, 0.0, -Xc[0]/(Z*Z)],
                          [0.0, 1.0/Z, -Xc[1]/(Z*Z)]],
                         dtype=K.dtype, device=K.device)
    A = torch.zeros((2,2), dtype=K.dtype, device=K.device); A[0,0]=fx; A[1,1]=fy
    return A @ dxdXc @ R

def _sigma_from_side(S: int, k: float) -> float:
    return max(0.5, (S-1)/(2.0*k))

def _lift_Simg_to_plane(S_img: torch.Tensor, J: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    Jp = torch.stack([e1, e2], dim=1)   # (3,2)
    Jplane = J @ Jp                     # (2,2)
    S_inv = torch.inverse(S_img + torch.eye(2, dtype=S_img.dtype, device=S_img.device)*1e-9)
    M = Jplane.T @ S_inv @ Jplane
    Sig = torch.inverse(M + torch.eye(2, dtype=M.dtype, device=M.device)*1e-12)
    return 0.5*(Sig+Sig.T) + torch.eye(2, dtype=Sig.dtype, device=Sig.device)*1e-12, Jplane

def _ellipse_pts_2d(Sig: torch.Tensor, n_std: float=3.0, N:int=360) -> np.ndarray:
    Sig = 0.5*(Sig+Sig.T)
    w,V = torch.linalg.eigh(Sig); w = torch.clamp(w, min=1e-12)
    axes = n_std*torch.sqrt(w)
    t = torch.linspace(0, 2*torch.pi, steps=N, device=Sig.device, dtype=Sig.dtype)
    circ = torch.stack([torch.cos(t), torch.sin(t)], dim=0)
    E = (V @ torch.diag(axes)) @ circ
    return E.detach().cpu().numpy(), V.detach().cpu().numpy(), axes.detach().cpu().numpy()

def visualize_tangent_plane_2d_with_epinormal(
    Ks: List[torch.Tensor], Rs: List[torch.Tensor], ts: List[torch.Tensor],
    images: List[torch.Tensor],
    X: torch.Tensor,
    views: Tuple[int,int],
    patch_side_S: int,
    patch_mult: float = 3.0,
    normalize_to: str = 'sigma',
    n_std: float = 3.0,
) -> Dict[str, object]:
    """
    2D tangent-plane visualization:
    - Build plane at X (baseline-normal).
    - Lift each view's image covariance to plane Σ_plane^i.
    - Map each view's epipolar-normal direction to the plane via J_plane^+.
    - Plot both ellipses on (u,v) axes, draw the two directions and annotate directional variances.
    """
    i1,i2 = views
    device, dtype = Ks[0].device, Ks[0].dtype
    C1, C2 = _camera_center(Rs[i1],ts[i1]), _camera_center(Rs[i2],ts[i2])
    n3 = _plane_normal_baseline(C1,C2,X)
    e1,e2 = _plane_basis(n3)
    Jp_stack = torch.stack([e1,e2], dim=1)  # (3,2)

    sigma = _sigma_from_side(patch_side_S, patch_mult)
    sigma_t = torch.tensor(sigma, dtype=dtype, device=device)

    # project points + fundamentals for epipolar lines
    x1_pix,_ = project_points(Ks[i1], Rs[i1], ts[i1], X)
    x2_pix,_ = project_points(Ks[i2], Rs[i2], ts[i2], X)
    x1_h = torch.tensor([x1_pix[0], x1_pix[1], 1.0], dtype=dtype, device=device)
    x2_h = torch.tensor([x2_pix[0], x2_pix[1], 1.0], dtype=dtype, device=device)
    F21 = compute_fundamental_from_poses(Ks[i2], Rs[i2], ts[i2], Ks[i1], Rs[i1], ts[i1])
    F12 = compute_fundamental_from_poses(Ks[i1], Rs[i1], ts[i1], Ks[i2], Rs[i2], ts[i2])
    l1 = F12 @ x2_h  # epipolar line in view1 from x2
    l2 = F21 @ x1_h  # epipolar line in view2 from x1
    d1 = l1[:2]; d1 = d1/(torch.norm(d1)+EPS)  # epipolar-normal (image)
    d2 = l2[:2]; d2 = d2/(torch.norm(d2)+EPS)

    out = {}

    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Tangent plane (u,v): lifted ellipses + epipolar-normal scales")

    colors = ['C0','C1']
    labels = [f"view {i1}", f"view {i2}"]
    vals = []

    for idx, (i, d_img) in enumerate([(i1,d1), (i2,d2)]):
        Ki,Ri,ti = Ks[i], Rs[i], ts[i]
        img = images[i]
        xi_pix = x1_pix if i==i1 else x2_pix

        # S_img and lift
        S_img = structure_tensor_cov_swapped(img, xi_pix, sigma_t, patch_mult=patch_mult, normalize_to=normalize_to)
        J = _J_proj(Ki,Ri,ti,X)
        Sigma_plane, Jplane = _lift_Simg_to_plane(S_img, J, e1, e2)

        # ellipse on plane
        E, V, axes = _ellipse_pts_2d(Sigma_plane, n_std=n_std, N=360)
        ax.plot(E[0], E[1], color=colors[idx], lw=2, label=f"{labels[idx]} ellipse")

        # map epipolar-normal to plane via pseudo-inverse
        Jpinv = torch.linalg.pinv(Jplane)  # (2,2)
        u = Jpinv @ d_img                   # in (e1,e2) coords
        u = u / (torch.norm(u)+EPS)
        # directional variance and std
        v = (u[None] @ Sigma_plane @ u[:,None]).item()
        std_dir = (v**0.5) * n_std
        vals.append((labels[idx], v))

        # draw direction arrow scaled by n_std*sqrt(v)
        u_np = u.detach().cpu().numpy()
        ax.arrow(0,0, u_np[0]*std_dir, u_np[1]*std_dir,
                 head_width=0.03*std_dir, head_length=0.05*std_dir,
                 color=colors[idx], length_includes_head=True, alpha=0.9)
        ax.text(u_np[0]*std_dir*1.05, u_np[1]*std_dir*1.05,
                f"{labels[idx]}: σ_dir={std_dir:.2f}", color=colors[idx], fontsize=9)

        # store
        out[f"Sigma_plane_{i}"] = Sigma_plane.detach().cpu()
        out[f"dir_std_{i}"] = std_dir
        out[f"dir_var_{i}"] = v

    # draw plane axes
    L = max(axes.max(), 3.0)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel("u (along e1)"); ax.set_ylabel("v (along e2)")
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)

    out["fig"] = fig
    return out