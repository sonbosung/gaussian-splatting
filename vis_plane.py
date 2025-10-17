import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Tuple, List, Optional, Dict

from vis_ellipse import (
    structure_tensor_cov_swapped,
    project_points,
    compute_fundamental_from_poses,
)

EPS = 1e-8

def camera_center_from_pose(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return -R.T @ t

def plane_normal_from_baseline(C1: torch.Tensor, C2: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    b = C2 - C1
    bb = (b @ b).clamp_min(EPS)
    tstar = ((X - C1) @ b) / bb
    P = C1 + tstar * b
    n = P - X
    n_norm = torch.norm(n) + EPS
    return n / n_norm

def orthonormal_basis_on_plane(n: torch.Tensor):
    a = torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device)
    if torch.abs(torch.dot(a, n)) > 0.9:
        a = torch.tensor([0.0, 1.0, 0.0], dtype=n.dtype, device=n.device)
    e1 = a - torch.dot(a, n) * n
    e1 = e1 / (torch.norm(e1) + EPS)
    e2 = torch.cross(n, e1)
    e2 = e2 / (torch.norm(e2) + EPS)
    return e1, e2

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

def sigma_from_side_support(S: int, k: float) -> float:
    return max(0.5, (S - 1) / (2.0 * k))

def lift_image_cov_to_plane(S_img: torch.Tensor, J: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    Jp = torch.stack([e1, e2], dim=1)  # (3,2)
    J_plane = J @ Jp                   # (2,2)
    S_inv = torch.inverse(S_img + torch.eye(2, dtype=S_img.dtype, device=S_img.device)*1e-9)
    M = J_plane.T @ S_inv @ J_plane
    Sigma_plane = torch.inverse(M + torch.eye(2, dtype=M.dtype, device=M.device)*1e-12)
    Sigma_plane = 0.5*(Sigma_plane + Sigma_plane.T) + torch.eye(2, dtype=Sigma_plane.dtype, device=Sigma_plane.device)*1e-12
    return Sigma_plane

def ellipse_points_from_cov_2d(Sigma: torch.Tensor, n_std: float = 3.0, num: int = 240) -> np.ndarray:
    Sigma = 0.5*(Sigma + Sigma.T)
    w, V = torch.linalg.eigh(Sigma)  # ascending
    w = torch.clamp(w, min=1e-12)
    axes = n_std * torch.sqrt(w)
    t = torch.linspace(0, 2*torch.pi, steps=num+1, device=Sigma.device)[:-1]
    circ = torch.stack([torch.cos(t), torch.sin(t)], dim=0)  # (2,N)
    E = (V @ torch.diag(axes)) @ circ  # (2,N)
    return E.detach().cpu().numpy()

def _clip_epipolar_to_image(line_abcp: torch.Tensor, H: int, W: int):
    """Return two points on the image border for line ax+by+c=0, or None if no intersection."""
    a, b, c = float(line_abcp[0]), float(line_abcp[1]), float(line_abcp[2])
    pts = []
    # x = 0..W-1 boundaries
    for x in [0.0, W-1.0]:
        if abs(b) > 1e-12:
            y = -(a*x + c)/b
            if 0.0 <= y <= H-1.0: pts.append((x,y))
    # y = 0..H-1 boundaries
    for y in [0.0, H-1.0]:
        if abs(a) > 1e-12:
            x = -(b*y + c)/a
            if 0.0 <= x <= W-1.0: pts.append((x,y))
    # deduplicate and pick two farthest
    if len(pts) < 2: return None
    best_d, best_pair = -1, None
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            dx, dy = pts[i][0]-pts[j][0], pts[i][1]-pts[j][1]
            d = dx*dx + dy*dy
            if d > best_d:
                best_d, best_pair = d, (pts[i], pts[j])
    return best_pair

def visualize_plane_and_lifted_ellipses(
    Ks: List[torch.Tensor], Rs: List[torch.Tensor], ts: List[torch.Tensor],
    images: List[torch.Tensor],
    X: torch.Tensor,
    views: Tuple[int,int],
    patch_side_S: int,
    patch_mult: float = 3.0,
    normalize_to: str = 'sigma',
    n_std_plane: float = 3.0,
    n_std_img: float = 6.0,
    draw_sigma_window: bool = True,
    draw_epipolar: bool = True,
) -> Dict[str, object]:
    """Visualize baseline-normal plane through X with ellipses lifted from two images.
    Renders TWO image subplots (one per view) + epipolar lines."""
    i1, i2 = views
    device, dtype = Ks[0].device, Ks[0].dtype

    # Camera centers & plane
    C1 = camera_center_from_pose(Rs[i1], ts[i1])
    C2 = camera_center_from_pose(Rs[i2], ts[i2])
    n3 = plane_normal_from_baseline(C1, C2, X)
    e1, e2 = orthonormal_basis_on_plane(n3)

    # Scales
    baseline_len = float(torch.norm(C2 - C1))
    L = max(1e-3, baseline_len * 0.4)  # plane extent
    normal_len = L * 0.6               # red arrow length

    sigma_val = sigma_from_side_support(patch_side_S, patch_mult)
    sigma_t = torch.tensor(sigma_val, dtype=dtype, device=device)

    # Figure: 3D + two image panels
    fig = plt.figure(figsize=(18,6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0])
    ax3d = fig.add_subplot(gs[0,0], projection='3d')
    axv1 = fig.add_subplot(gs[0,1])
    axv2 = fig.add_subplot(gs[0,2])

    # 3D plane surface
    us = np.linspace(-L, L, 20)
    vs = np.linspace(-L, L, 20)
    UU, VV = np.meshgrid(us, vs, indexing='xy')
    P = X.detach().cpu().numpy().reshape(3,1,1) + \
        e1.detach().cpu().numpy().reshape(3,1,1) * UU.reshape(1, *UU.shape) + \
        e2.detach().cpu().numpy().reshape(3,1,1) * VV.reshape(1, *VV.shape)
    ax3d.plot_surface(P[0], P[1], P[2], alpha=0.25, color='#8ecae6', edgecolor='none')

    # Cameras & baseline
    C1_np = C1.detach().cpu().numpy(); C2_np = C2.detach().cpu().numpy()
    ax3d.scatter([C1_np[0]],[C1_np[1]],[C1_np[2]], color='tab:blue', s=40, label=f"view {i1}")
    ax3d.scatter([C2_np[0]],[C2_np[1]],[C2_np[2]], color='tab:orange', s=40, label=f"view {i2}")
    ax3d.plot([C1_np[0], C2_np[0]],[C1_np[1], C2_np[1]],[C1_np[2], C2_np[2]], color='g', linewidth=2)

    # Point X & normal
    X_np = X.detach().cpu().numpy()
    ax3d.scatter([X_np[0]],[X_np[1]],[X_np[2]], color='k', s=30)
    n_end = (X + n3 * normal_len).detach().cpu().numpy()
    ax3d.plot([X_np[0], n_end[0]],[X_np[1], n_end[1]],[X_np[2], n_end[2]], color='r', linewidth=2)

    try:
        ax3d.set_box_aspect([1,1,1])
    except Exception:
        pass
    ax3d.set_title("Plane through X (baseline-normal) + lifted ellipses")
    ax3d.legend(loc='upper left')

    def draw_view(ax, img, xi_pix, S_img, color, label, epi_line=None):
        if img.ndim == 2:
            im_np = img.detach().cpu().numpy()
        else:
            im_np = img[0,0].detach().cpu().numpy()
        H, W = im_np.shape[:2]
        ax.imshow(im_np, origin='upper', cmap='gray')

        # ellipse
        w_img, V_img = torch.linalg.eigh(0.5*(S_img+S_img.T))
        w_img = torch.clamp(w_img, min=1e-12)
        axes = n_std_img * torch.sqrt(w_img)
        t = torch.linspace(0, 2*torch.pi, steps=361, device=S_img.device)[:-1]
        circ = torch.stack([torch.cos(t), torch.sin(t)], dim=0)
        Eim = (V_img @ torch.diag(axes)) @ circ
        Eim_np = Eim.detach().cpu().numpy()
        cx, cy = float(xi_pix[0]), float(xi_pix[1])
        ax.plot(Eim_np[0,:]+cx, Eim_np[1,:]+cy, color=color, linewidth=2, label=label)

        # epipolar line
        if epi_line is not None:
            seg = _clip_epipolar_to_image(epi_line, H, W)
            if seg is not None:
                (x1,y1),(x2,y2) = seg
                ax.plot([x1,x2],[y1,y2], linestyle='--', color='w', linewidth=1.5, alpha=0.9)

        if draw_sigma_window:
            import matplotlib.patches as patches
            rad = patch_mult * float(sigma_val)
            ax.add_patch(patches.Circle((cx, cy), radius=rad, fill=False, linestyle='--', edgecolor=color, linewidth=1.5))

        ax.set_xlim([0, W-1]); ax.set_ylim([H-1, 0])
        ax.set_aspect('equal')
        ax.legend(loc='upper right')

    debug = {"centers": [], "S_img": [], "Sigma_plane": []}
    colors = ['C0', 'C1']
    labels = [f"view {i1}", f"view {i2}"]

    # projections of X in both views
    x1_pix, _ = project_points(Ks[i1], Rs[i1], ts[i1], X)
    x2_pix, _ = project_points(Ks[i2], Rs[i2], ts[i2], X)
    x1_h = torch.tensor([x1_pix[0], x1_pix[1], 1.0], dtype=dtype, device=device)
    x2_h = torch.tensor([x2_pix[0], x2_pix[1], 1.0], dtype=dtype, device=device)

    # fundamental matrices between the two views
    F21 = compute_fundamental_from_poses(Ks[i2], Rs[i2], ts[i2], Ks[i1], Rs[i1], ts[i1])  # from view1 to view2
    F12 = compute_fundamental_from_poses(Ks[i1], Rs[i1], ts[i1], Ks[i2], Rs[i2], ts[i2])  # from view2 to view1
    l2 = F21 @ x1_h  # epipolar line in view2 from x1
    l1 = F12 @ x2_h  # epipolar line in view1 from x2

    # For each view, compute measured S_img and lifted ellipse on plane
    for idx, (axv, i, epi_line) in enumerate(zip([axv1, axv2], [i1, i2], [l1, l2])):
        Ki, Ri, ti = Ks[i], Rs[i], ts[i]
        img = images[i]
        xi_pix = x1_pix if i==i1 else x2_pix
        J = project_jacobian(Ki, Ri, ti, X)

        # measured image covariance
        S_img = structure_tensor_cov_swapped(
            image=img, center_xy=xi_pix, sigma=sigma_t,
            patch_mult=patch_mult, normalize_to=normalize_to
        )

        # lift to plane and draw in 3D
        Sigma_plane = lift_image_cov_to_plane(S_img, J, e1, e2)
        Euv = ellipse_points_from_cov_2d(Sigma_plane, n_std=n_std_plane, num=360)
        pts3 = (X[:,None] +
                e1[:,None]*torch.tensor(Euv[0], dtype=dtype, device=device) +
                e2[:,None]*torch.tensor(Euv[1], dtype=dtype, device=device)).detach().cpu().numpy()
        ax3d.plot(pts3[0], pts3[1], pts3[2], label=labels[idx], color=colors[idx], linewidth=2)

        # draw on its own image
        draw_view(axv, img, xi_pix, S_img, colors[idx], labels[idx], epi_line=epi_line if draw_epipolar else None)

        debug["centers"].append(xi_pix.detach().cpu())
        debug["S_img"].append(S_img.detach().cpu())
        debug["Sigma_plane"].append(Sigma_plane.detach().cpu())

    plt.tight_layout()
    return {"fig": fig, **debug}

def visualize_planes_and_lifted_ellipses_multi(
    Ks: List[torch.Tensor], Rs: List[torch.Tensor], ts: List[torch.Tensor],
    images: List[torch.Tensor],
    X_list: List[torch.Tensor],
    views: Tuple[int,int],
    patch_side_S: int,
    patch_mult: float = 3.0,
    normalize_to: str = 'sigma',
    n_std_plane: float = 3.0,
    n_std_img: float = 6.0,
    draw_sigma_window: bool = False,
    draw_epipolar: bool = False,
    cmap_name: str = 'tab10',
) -> Dict[str, object]:
    """
    Accumulate multiple points X_k in a single figure.
    - Left: 3D, for each X_k draw its baseline-normal plane and lifted ellipses (two views).
    - Right two panels: overlay all per-view ellipses on each image.
    - Annotate principal-axis lengths (a,b) on the plane for each lifted ellipse.
    """
    i1, i2 = views
    device, dtype = Ks[0].device, Ks[0].dtype

    # Figure: 3D + two image panels
    fig = plt.figure(figsize=(20,7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0])
    ax3d = fig.add_subplot(gs[0,0], projection='3d')
    axv1 = fig.add_subplot(gs[0,1])
    axv2 = fig.add_subplot(gs[0,2])

    # Initialize image panels
    for ax, i in [(axv1, i1), (axv2, i2)]:
        img = images[i]
        im_np = img.detach().cpu().numpy() if img.ndim==2 else img[0,0].detach().cpu().numpy()
        ax.imshow(im_np, origin='upper', cmap='gray')
        H, W = im_np.shape[:2]
        ax.set_xlim([0, W-1]); ax.set_ylim([H-1, 0]); ax.set_aspect('equal')

    # Precompute fundamentals for epipolar lines
    if draw_epipolar:
        from vis_ellipse import compute_fundamental_from_poses
        F21 = compute_fundamental_from_poses(Ks[i2], Rs[i2], ts[i2], Ks[i1], Rs[i1], ts[i1])  # from 1->2
        F12 = compute_fundamental_from_poses(Ks[i1], Rs[i1], ts[i1], Ks[i2], Rs[i2], ts[i2])  # from 2->1

    # Global colors per point
    cmap = plt.get_cmap(cmap_name)
    sigma_val = max(0.5, (patch_side_S - 1) / (2.0 * patch_mult))
    sigma_t = torch.tensor(sigma_val, dtype=dtype, device=device)

    # Draw cameras and baseline once (using the first point's plane to set scale)
    C1 = -Rs[i1].T @ ts[i1]; C2 = -Rs[i2].T @ ts[i2]
    C1_np, C2_np = C1.detach().cpu().numpy(), C2.detach().cpu().numpy()
    ax3d.scatter([C1_np[0]],[C1_np[1]],[C1_np[2]], color='tab:blue', s=40, label=f"view {i1}")
    ax3d.scatter([C2_np[0]],[C2_np[1]],[C2_np[2]], color='tab:orange', s=40, label=f"view {i2}")
    ax3d.plot([C1_np[0], C2_np[0]],[C1_np[1], C2_np[1]],[C1_np[2], C2_np[2]], color='g', linewidth=2)

    # Iterate all points
    for k, X in enumerate(X_list):
        color = cmap(k % 10)
        # plane
        n3 = plane_normal_from_baseline(C1, C2, X)
        e1, e2 = orthonormal_basis_on_plane(n3)
        baseline_len = float(torch.norm(C2 - C1))
        L = max(1e-3, baseline_len * 0.25)  # slightly smaller planes for clutter
        normal_len = L * 0.5

        # plane surface
        us = np.linspace(-L, L, 10)
        vs = np.linspace(-L, L, 10)
        UU, VV = np.meshgrid(us, vs, indexing='xy')
        Psurf = X.detach().cpu().numpy().reshape(3,1,1) + \
                e1.detach().cpu().numpy().reshape(3,1,1) * UU.reshape(1,*UU.shape) + \
                e2.detach().cpu().numpy().reshape(3,1,1) * VV.reshape(1,*VV.shape)
        ax3d.plot_surface(Psurf[0], Psurf[1], Psurf[2], alpha=0.15, color=color, edgecolor='none')

        # X and normal
        Xn = X.detach().cpu().numpy()
        ax3d.scatter([Xn[0]],[Xn[1]],[Xn[2]], color=color, s=18)
        n_end = (X + n3 * normal_len).detach().cpu().numpy()
        ax3d.plot([Xn[0], n_end[0]],[Xn[1], n_end[1]],[Xn[2], n_end[2]], color=color, linewidth=1.8)

        # Per-view ellipses (image + plane)
        for ax_img, i in [(axv1, i1), (axv2, i2)]:
            Ki, Ri, ti = Ks[i], Rs[i], ts[i]
            img = images[i]
            xi_pix, _ = project_points(Ki, Ri, ti, X)
            J = project_jacobian(Ki, Ri, ti, X)

            # image covariance & draw
            S_img = structure_tensor_cov_swapped(img, xi_pix, sigma_t, patch_mult=patch_mult, normalize_to=normalize_to)
            # ellipse in pixel coords
            w_img, V_img = torch.linalg.eigh(0.5*(S_img+S_img.T))
            w_img = torch.clamp(w_img, min=1e-12)
            axes_img = n_std_img * torch.sqrt(w_img)
            t = torch.linspace(0, 2*torch.pi, steps=361, device=S_img.device)[:-1]
            circ = torch.stack([torch.cos(t), torch.sin(t)], dim=0)
            Eim = (V_img @ torch.diag(axes_img)) @ circ
            Eim_np = Eim.detach().cpu().numpy()
            cx, cy = float(xi_pix[0]), float(xi_pix[1])
            ax_img.plot(Eim_np[0,:]+cx, Eim_np[1,:]+cy, color=color, linewidth=1.8)

            # epipolar
            if draw_epipolar:
                x_h = torch.tensor([cx, cy, 1.0], dtype=dtype, device=device)
                if i == i1:
                    l_other = F21 @ x_h
                else:
                    l_other = F12 @ x_h
                H, W = ax_img.images[0].get_array().shape[:2]
                seg = _clip_epipolar_to_image(l_other, H, W)
                if seg is not None:
                    (x1,y1),(x2,y2) = seg
                    ax_img.plot([x1,x2],[y1,y2], linestyle='--', color='w', linewidth=1.0, alpha=0.7)

            # lift to plane and draw + annotate axes
            Sigma_plane = lift_image_cov_to_plane(S_img, J, e1, e2)
            Euv = ellipse_points_from_cov_2d(Sigma_plane, n_std=n_std_plane, num=240)
            pts3 = (X[:,None] + e1[:,None]*torch.tensor(Euv[0], dtype=dtype, device=device)
                              + e2[:,None]*torch.tensor(Euv[1], dtype=dtype, device=device)).detach().cpu().numpy()
            ax3d.plot(pts3[0], pts3[1], pts3[2], color=color, linewidth=2.2)

            # principal axis lengths (a,b) on plane
            w_pl, V_pl = torch.linalg.eigh(0.5*(Sigma_plane+Sigma_plane.T))
            a_len = float(n_std_plane*torch.sqrt(torch.clamp(w_pl[1], min=1e-12)))  # major
            b_len = float(n_std_plane*torch.sqrt(torch.clamp(w_pl[0], min=1e-12)))  # minor
            # Put small text near point X with offset along e1/e2 to avoid overlap
            pt_anno = (X + 0.1*L*e1 + 0.05*L*e2).detach().cpu().numpy()
            ax3d.text(pt_anno[0], pt_anno[1], pt_anno[2],
                      f'a={a_len:.2f}, b={b_len:.2f}', color=color, fontsize=8)

    try:
        ax3d.set_box_aspect([1,1,1])
    except Exception:
        pass
    ax3d.set_title("Multiple points: baseline-normal planes + lifted ellipses (annotated a,b)")
    ax3d.legend(loc='upper left')
    plt.tight_layout()
    return {"fig": fig}