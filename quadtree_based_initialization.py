import torch
import numpy as np
import os
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2

from augmentation_new import Augmentor
from scene.gaussian_model import GaussianModel
from utils.general_utils import build_scaling_rotation
from utils.sh_utils import SH2RGB
from gaussian_renderer import render, network_gui
import sys
import traceback
import sys
import traceback
import pdb

def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    print("\nException occurred! Launching debugger...\n")
    pdb.post_mortem(tb)

sys.excepthook = info

class GaussianInitializer:
    """
    Initializes Gaussian attributes (features, scaling, rotation) based on 2D covariance
    information derived from a quadtree decomposition of the input images.
    """
    def __init__(self, gaussians, cameras, dataset, n_steps=100000, lr=1e-3, batch_size=256, pipe=None, background=None, gs_dataset=None, opt=None, args=None):
        self.gaussians = gaussians
        self.cameras = cameras
        self.dataset = dataset
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.world_view_transform = torch.stack([camera.world_view_transform.transpose(0, 1) for camera in cameras])
        self.proj_matrix = torch.stack([camera.projection_matrix.transpose(0, 1) for camera in cameras])
        self.image_width_height = torch.stack([torch.tensor([camera.image_width, camera.image_height], dtype=torch.float32, device="cuda") for camera in cameras])
        
        print(f"{len(dataset.p3d_image_id_matrix)} points in dataset")
        self.camera_ids_to_indices()
        
        self.dataset.corresponding_cov_matrix = torch.tensor(np.stack(self.dataset.corresponding_cov_matrix), dtype=torch.float32, device="cuda")
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.args = args
        self.pipe = pipe
        self.background = background
        self.iteration = 0
        self.gs_dataset = gs_dataset
        self.opt = opt
        self.mode = "sfm"  # Initial mode is 'sfm', can be switched to 'augmentation'

        # self.image_name_gsid = {camera.image_name: idx for idx, camera in enumerate(cameras)}
        # self.dataset.convert_image_ids_gs(self.image_name_gsid)

    def camera_ids_to_indices(self):
        colmap_image_id_names = {id: colmap_image.name for (id, colmap_image) in self.dataset.augmentor.colmap_images.items()}
        camera_name_index = {camera.image_name: idx for idx, camera in enumerate(self.cameras)}
        colmap_image_id_to_camera_index = {}
        for id, name in colmap_image_id_names.items():
            if name in camera_name_index.keys():
                colmap_image_id_to_camera_index[id] = camera_name_index[name]

        to_delete = []
        for i, (_, colmap_image_id, _, _) in enumerate(self.dataset.p3d_image_id_matrix):
            if colmap_image_id not in colmap_image_id_to_camera_index.keys():
                to_delete.append(i)
            else:
                self.dataset.p3d_image_id_matrix[i][1] = colmap_image_id_to_camera_index[colmap_image_id]

        print(f"Filtered {len(to_delete)} entries corresponding to test cameras.")
        self.colmap_image_id_to_camera_index = colmap_image_id_to_camera_index

    def _verify_jacobian_numerically(self, pmat, xyz, K, Rt, epsilon=1e-6):
        """
        Verifies the analytical Jacobian computation against a numerical one using finite differences.
        This is a debugging function and should be used sparingly.
        """
        print("DEBUG: Verifying Jacobian computation numerically...")
        
        # Use batch-style inputs for one element
        xyz_batch = xyz.unsqueeze(0)
        K_batch = K.unsqueeze(0)
        Rt_batch = Rt.unsqueeze(0)

        # Analytical Jacobian
        J_analytical = self.compute_jacobian_batch(xyz_batch, K_batch, Rt_batch).squeeze(0)

        # Numerical Jacobian
        J_numerical = torch.zeros_like(J_analytical)
        for i in range(3):
            # Perturb along the i-th dimension of xyz
            xyz_plus = xyz.clone()
            xyz_plus[i] += epsilon
            xyz_minus = xyz.clone()
            xyz_minus[i] -= epsilon

            # Compute projected points
            Xh_plus = torch.cat([xyz_plus, torch.ones(1, device=xyz.device)])
            xy_plus = (pmat @ Xh_plus)[:2] / (pmat @ Xh_plus)[2]

            Xh_minus = torch.cat([xyz_minus, torch.ones(1, device=xyz.device)])
            xy_minus = (pmat @ Xh_minus)[:2] / (pmat @ Xh_minus)[2]
            
            # Central difference
            grad = (xy_plus - xy_minus) / (2 * epsilon)
            J_numerical[:, i] = grad

        diff = torch.norm(J_analytical - J_numerical)
        print(f"DEBUG: Jacobian verification complete. Difference norm: {diff.item()}")
        if diff > 1e-4: # A reasonably small threshold
            print("WARNING: High difference between analytical and numerical Jacobian. Check implementation.")
            print("Analytical:\n", J_analytical)
            print("Numerical:\n", J_numerical)
        
        return diff

    def run(self):
        """
        Runs the optimization process for the SfM points.
        """
        assert self.mode == "sfm", "run() should only be called in sfm mode"
        
        # --- DEBUG: Verify Jacobian for one example ---
        if self.iteration == 0:
            try:
                sample_x, _ = next(iter(self.dataloader))
                sample_point_idx, sample_camera_id = sample_x
                p_idx, c_idx = sample_point_idx[0], sample_camera_id[0]
                
                Rt = self.world_view_transform[c_idx]
                proj_matrix = self.proj_matrix[c_idx]
                image_width = self.image_width_height[c_idx][0]
                image_height = self.image_width_height[c_idx][1]
                intrinsic_matrix = self.P_gl_to_P_cv(proj_matrix.unsqueeze(0), image_width, image_height).squeeze(0)
                pmat = intrinsic_matrix @ Rt[:3, :4]
                xyz = self.gaussians._xyz[p_idx]
                self._verify_jacobian_numerically(pmat, xyz, intrinsic_matrix, Rt)
            except Exception as e:
                print(f"DEBUG: Could not run Jacobian verification. Error: {e}")
        # --- End of DEBUG ---

        params = [
            {'params': [self.gaussians._features_dc], 'lr': self.lr, "name": "features_dc"},
            {'params': [self.gaussians._scaling], 'lr': self.lr, "name": "scaling"},
            {'params': [self.gaussians._rotation], 'lr': self.lr, "name": "rotation"}
        ]
        optimizer = torch.optim.Adam(params)
        
        dataloader_iter = iter(self.dataloader)
        progress_bar = tqdm(range(self.n_steps), desc="Initializing SfM points")

        for step in progress_bar:
            self._handle_gui_connection()

            try:
                batch_x, batch_y = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                batch_x, batch_y = next(dataloader_iter)

            batch_point_idx, batch_camera_id = batch_x
            batch_gt_cov = batch_y

            if torch.isnan(batch_gt_cov).any() or torch.isinf(batch_gt_cov).any():
                print("Warning: NaN or Inf detected in ground truth covariance. Skipping batch.")
                continue

            loss = self._compute_loss(batch_point_idx, batch_camera_id, batch_gt_cov)

            if torch.isnan(loss):
                print(f"NaN loss detected at step {step}. Skipping update.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})

        for i in range(self.gaussians._xyz.shape[0]):
            index = torch.tensor([i], device=self.gaussians._xyz.device)
            self.dataset.colmap3Dpoints[i].cov3Dmatrix = self.gaussians.get_covariance_slice(index).clone().detach().cpu().numpy()

        colmap3Dpoints = self.dataset.colmap3Dpoints
        with open(os.path.join(self.dataset.image_path, "colmap3Dpoints.pkl"), "wb") as f:
            pickle.dump(colmap3Dpoints, f)
            print(f"Saved colmap3Dpoints to {os.path.join(self.dataset.image_path, 'colmap3Dpoints.pkl')}")

    def run_for_augmentation(self):
        """
        Runs the optimization process for the augmented points.
        """
        assert self.mode == "augmentation", "run_for_augmentation() should only be called in augmentation mode"
        
        self.n_steps = 50000
        params = [
            {'params': [self.gaussians._features_dc], 'lr': self.lr, "name": "features_dc"},
            {'params': [self.gaussians._scaling], 'lr': self.lr, "name": "scaling"},
            {'params': [self.gaussians._rotation], 'lr': self.lr, "name": "rotation"},
            {'params': [self.gaussians._xyz], 'lr': self.lr, "name": "xyz"}
        ]
        optimizer = torch.optim.Adam(params)
        
        self.dataset.p3d_image_ids = [self.colmap_image_id_to_camera_index[image_id] for image_id in self.dataset.p3d_image_ids]
        dataloader_iter = iter(self.dataloader)
        progress_bar = tqdm(range(self.n_steps), desc="Initializing augmented points")

        for step in progress_bar:
            try:
                batch_x, batch_y = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                batch_x, batch_y = next(dataloader_iter)
        
            batch_point_idx, batch_camera_id = batch_x
            batch_gt_cov = batch_y

            if torch.isnan(batch_gt_cov).any() or torch.isinf(batch_gt_cov).any():
                print("Warning: NaN or Inf detected in ground truth covariance. Skipping batch.")
                continue

            loss = self._compute_loss(batch_point_idx, batch_camera_id, batch_gt_cov)

            if torch.isnan(loss):
                print(f"NaN loss detected at step {step}. Skipping update.")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})

    def _compute_loss(self, point_indices, camera_indices, gt_cov):
        """Computes the Frobenius norm loss between predicted and ground truth 2D covariances."""
        pred_xyz = self.gaussians._xyz[point_indices]
        
        # Projection transformation
        Rt = self.world_view_transform[camera_indices]
        proj_matrix = self.proj_matrix[camera_indices]
        image_width = self.image_width_height[camera_indices][:, 0]
        image_height = self.image_width_height[camera_indices][:, 1]
        intrinsic_matrix = self.P_gl_to_P_cv(proj_matrix, image_width, image_height)
        pmat = torch.bmm(intrinsic_matrix, Rt[:, :3, :4])
        
        # --- DEBUG: Check for numerical stability ---
        B = pred_xyz.shape[0]
        ones = torch.ones((B, 1), dtype=pred_xyz.dtype, device=pred_xyz.device)
        Xh = torch.cat([pred_xyz, ones], dim=1)
        w = torch.sum(pmat[:, 2] * Xh, dim=1)
        if torch.any(w < 1e-3):
            print(f"DEBUG: Warning: Small w detected (min: {w.min().item()}). Points may be near/behind camera plane.")
        # import pdb; pdb.set_trace()  # Debugging breakpoint
        J_batch = self.compute_jacobian_batch(pred_xyz, intrinsic_matrix, Rt)
        J_norm = torch.norm(J_batch, p='fro', dim=(1, 2))
        
        
        # --- End of DEBUG ---

        # Predict 2D covariance
        pred_cov = self.gaussians.get_covariance_slice(point_indices)
        pred_cov3d = self.uppertri_to_symm(pred_cov)
        pred_cov2d = self.project_cov3d_to_2d_batch(pred_cov3d, J_batch)
        # kl_loss = self.kl_divergence_2d_gaussian(None, gt_cov, None, pred_cov2d)
        airm_distance = self.airm_distance(gt_cov, pred_cov2d)
        size_reg_loss = self.size_regularization_logdet(pred_cov2d, gt_cov, weight=0.1)
        eigenvalue_max = self.compute_max_eigenvalue(pred_cov2d)
        # longaxisloss = torch.mean((eigenvalue_max - 1.0) ** 2)

        # # Add a small identity matrix for numerical stability
        # epsilon = 1e-6
        # identity = torch.eye(pred_cov2d.size(-1), device=pred_cov2d.device) * epsilon
        # pred_cov2d += identity
        # gt_cov += identity
        
        # # Compute loss
        # # frobenius_loss = self.frobenius_norm_batch(gt_cov, pred_cov2d).mean()

        # det_gt = torch.det(gt_cov)
        # det_pred = torch.det(pred_cov2d)
        # det_ratio = det_pred / (det_gt + 1e-8)

        # scale_gt = torch.sqrt(det_gt + 1e-8)
        # scale_pred = torch.sqrt(det_pred + 1e-8)
        # gt_cov_normalized = gt_cov / scale_gt.unsqueeze(-1).unsqueeze(-1)
        # pred_cov2d_normalized = pred_cov2d / scale_pred.unsqueeze(-1).unsqueeze(-1)

        # shape_loss = self.frobenius_norm_batch(gt_cov_normalized, pred_cov2d_normalized).mean()
        # # scale_loss = torch.mean(torch.abs(torch.log(scale_gt + 1e-8) - torch.log(scale_pred + 1e-8)))
        # det_loss = torch.mean(torch.abs(det_ratio - 1.0))
        
        # # return shape_loss + scale_loss
        # if torch.rand(1) < 0.01:
        #     print(f"DEBUG: Step {self.iteration}, Shape Loss: {shape_loss.item()}, Det Loss: {det_loss.item()}")
        # return shape_loss + 0.1 * det_loss
        # return kl_loss + longaxisloss
        return airm_distance.mean() + size_reg_loss
    
    def size_regularization_logdet(self, sigma_pred, sigma_gt, weight=1.0):
        logdet_pred = torch.logdet(sigma_pred)
        logdet_gt = torch.logdet(sigma_gt)
        reg = torch.relu(logdet_gt - logdet_pred)  # Only penalize when predicted is smaller
        return weight * reg.mean()


    def _handle_gui_connection(self):
        """Handles connection and communication with the GUI."""
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifier=scaling_modifier, use_trained_exp=self.gs_dataset.train_test_exp, separate_sh=False)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, self.gs_dataset.source_path)
                if do_training and ((self.iteration < int(self.opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

    def augmentation_mode(self):
        """Switches the initializer to augmentation mode."""
        self.mode = "augmentation"
        self.dataset.augmentation_mode()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def P_gl_to_P_cv(self, P_gl, width, height):
        """Converts an OpenGL projection matrix to a OpenCV intrinsic matrix."""
        B = P_gl.shape[0]
        fx = P_gl[:, 0, 0] * (width / 2.0)
        fy = P_gl[:, 1, 1] * (height / 2.0)
        cx = (1 + P_gl[:, 0, 2]) * (width / 2.0)
        cy = (1 - P_gl[:, 1, 2]) * (height / 2.0)

        K = torch.zeros((B, 3, 3), device=P_gl.device, dtype=P_gl.dtype)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        return K

    def uppertri_to_symm(self, upper_triangular):
        """Converts a 6D upper triangular vector to a 3x3 symmetric matrix."""
        B = upper_triangular.shape[0]
        symm_matrix = torch.zeros((B, 3, 3), device=upper_triangular.device, dtype=upper_triangular.dtype)
        
        # Upper triangular 형태: [xx, xy, xz, yy, yz, zz]
        symm_matrix[:, 0, 0] = upper_triangular[:, 0]  # xx
        symm_matrix[:, 0, 1] = upper_triangular[:, 1]  # xy
        symm_matrix[:, 0, 2] = upper_triangular[:, 2]  # xz
        symm_matrix[:, 1, 1] = upper_triangular[:, 3]  # yy
        symm_matrix[:, 1, 2] = upper_triangular[:, 4]  # yz
        symm_matrix[:, 2, 2] = upper_triangular[:, 5]  # zz
        
        # 대칭성 보장 (대각선 제외)
        symm_matrix[:, 1, 0] = symm_matrix[:, 0, 1]  # xy
        symm_matrix[:, 2, 0] = symm_matrix[:, 0, 2]  # xz
        symm_matrix[:, 2, 1] = symm_matrix[:, 1, 2]  # yz
        
        return symm_matrix
    
    def compute_max_eigenvalue(self, covmat2D):
        a = covmat2D[:, 0, 0]
        c = covmat2D[:, 0, 1]
        b = covmat2D[:, 1, 1]

        lambda_max = (a + b) / 2 + torch.sqrt(((a - b) / 2) ** 2 + c ** 2)
        return lambda_max

    # def camera_ids_to_indices(self):
    #     """Maps COLMAP image IDs to camera indices used in the training scene."""
    #     colmap_image_id_names = {id: colmap_image.name for id, colmap_image in self.dataset.augmentor.colmap_images.items()}
    #     camera_name_index = {camera.image_name: idx for idx, camera in enumerate(self.cameras)}
        
    #     self.colmap_image_id_to_camera_index = {}
    #     for id, name in colmap_image_id_names.items():
    #         if name in camera_name_index:
    #             self.colmap_image_id_to_camera_index[id] = camera_name_index[name]

    #     valid_entries = []
    #     for entry in self.dataset.p3d_image_id_matrix:
    #         colmap_image_id = entry[1]
    #         if colmap_image_id in self.colmap_image_id_to_camera_index:
    #             entry[1] = self.colmap_image_id_to_camera_index[colmap_image_id]
    #             valid_entries.append(entry)
        
    #     print(f"Filtered {len(self.dataset.p3d_image_id_matrix) - len(valid_entries)} entries corresponding to test cameras.")
    #     self.dataset.p3d_image_id_matrix = valid_entries

    def project_cov3d_to_2d_batch(self, cov3d_batch, J_batch):
        """Projects a batch of 3D covariance matrices to 2D."""
        cov2d_batch = torch.bmm(torch.bmm(J_batch, cov3d_batch), J_batch.transpose(1, 2))
        return cov2d_batch

    def compute_jacobian_batch(self, xyz_batch, K_batch, Rt_batch):
        """Computes the Jacobian of the projection function for a batch of points."""
        B = xyz_batch.shape[0]
        
        # World to camera coordinates
        R_batch = Rt_batch[:, :3, :3]
        t_batch = Rt_batch[:, :3, 3]
        
        xyz_cam_batch = torch.bmm(R_batch, xyz_batch.unsqueeze(-1)) + t_batch.unsqueeze(-1)
        xyz_cam_batch = xyz_cam_batch.squeeze(-1)
        
        x_cam = xyz_cam_batch[:, 0]
        y_cam = xyz_cam_batch[:, 1]
        z_cam = xyz_cam_batch[:, 2]
        
        # Avoid division by zero
        eps = 1e-8
        z_safe = torch.where(torch.abs(z_cam) < eps, torch.full_like(z_cam, eps), z_cam)
        
        inv_z = 1.0 / z_safe
        inv_z_sq = inv_z * inv_z
        
        # Intrinsics
        fx = K_batch[:, 0, 0]
        fy = K_batch[:, 1, 1]
        
        # Jacobian of pinhole projection wrt camera coordinates
        J_pinhole = torch.zeros((B, 2, 3), device=xyz_batch.device, dtype=xyz_batch.dtype)
        J_pinhole[:, 0, 0] = fx * inv_z
        J_pinhole[:, 0, 2] = -fx * x_cam * inv_z_sq
        J_pinhole[:, 1, 1] = fy * inv_z
        J_pinhole[:, 1, 2] = -fy * y_cam * inv_z_sq
        
        # Full Jacobian is J_pinhole @ R
        J_batch = torch.bmm(J_pinhole, R_batch)
        
        return J_batch

    def frobenius_norm_batch(self, cov1_batch, cov2_batch):
        """Computes the Frobenius norm of the difference between two batches of matrices."""
        diff = cov1_batch - cov2_batch
        return torch.norm(diff, p='fro', dim=(1, 2))
    
    def kl_divergence_2d_gaussian(self, mu1, cov1, mu2, cov2):
        """2D Gaussian KL divergence: KL(P||Q)"""
        k = 2
        
        # 수치적 안정성을 위한 처리
        det_cov1 = torch.det(cov1)
        det_cov2 = torch.det(cov2)
        
        # 음수 determinant 방지
        det_cov1 = torch.clamp(det_cov1, min=1e-8)
        det_cov2 = torch.clamp(det_cov2, min=1e-8)
        
        log_det_term = torch.log(det_cov2) - torch.log(det_cov1)
        
        # Trace term: tr(cov2^-1 @ cov1)
        try:
            cov2_inv = torch.inverse(cov2)
            trace_term = torch.diagonal(torch.bmm(cov2_inv, cov1), dim1=1, dim2=2).sum(dim=1)
        except:
            # Fallback: pseudo-inverse
            cov2_inv = torch.pinverse(cov2)
            trace_term = torch.diagonal(torch.bmm(cov2_inv, cov1), dim1=1, dim2=2).sum(dim=1)
        
        # Mean difference term (여기서는 0으로 가정)

        # mu_diff = mu2 - mu1
            # mu_diff = 0
            # if torch.norm(mu_diff) > 1e-8:
            #     quad_term = torch.sum(mu_diff.unsqueeze(-1) * torch.bmm(cov2_inv, mu_diff.unsqueeze(-1)), dim=(-2, -1))
            # else:
            #     quad_term = torch.zeros_like(trace_term)

        kl = 0.5 * (log_det_term + trace_term - k)
        return kl.mean()
    
    def matrix_sqrt_inv(self, mat):
        # Compute the inverse square root of a SPD matrix (batch of 2x2)
        eigvals, eigvecs = torch.linalg.eigh(mat)  # eigvals: (B, 2), eigvecs: (B, 2, 2)
        sqrt_inv = eigvecs @ torch.diag_embed(1.0 / torch.sqrt(eigvals)) @ eigvecs.transpose(-2, -1)
        return sqrt_inv

    def matrix_log(self, mat):
        # Compute the matrix logarithm of a SPD matrix (batch of 2x2)
        eigvals, eigvecs = torch.linalg.eigh(mat)
        log_eigvals = torch.log(eigvals)
        return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-2, -1)

    def airm_distance(self, sigma1: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        """
        Compute the Affine-Invariant Riemannian Metric between batches of SPD 2x2 matrices.

        Args:
            sigma1: Tensor of shape (B, 2, 2)
            sigma2: Tensor of shape (B, 2, 2)

        Returns:
            Tensor of shape (B,) with AIRM distances
        """
        sigma1_inv_sqrt = self.matrix_sqrt_inv(sigma1)
        middle = sigma1_inv_sqrt @ sigma2 @ sigma1_inv_sqrt
        log_middle = self.matrix_log(middle)
        frob_norm = torch.linalg.norm(log_middle, dim=(1, 2))
        return frob_norm

class QuadtreeInitDataset(Dataset):
    """
    Dataset for quadtree-based initialization. It generates 2D covariance matrices
    from image gradients.
    """
    def __init__(self, image_path, colmap_path):
        self.image_path = image_path
        self.colmap_path = colmap_path
        self._load_or_create_augmentor()
        
        self.image_scale = int(image_path.split("/")[-1].split("_")[-1]) if "images_" in image_path else 1
        self.scale = 1
        print(f"Scale set to {self.scale}! This is used to scale the covariance matrix.")
        
        # self._compute_image_gradients()
        # self.create_p3d_image_id_matrix()
        # self.mode = "sfm"
        # self.n_colmap_points = len(self.augmentor.colmap_points3D)

    def _load_or_create_augmentor(self):
        """Loads a pre-existing augmentor or creates a new one."""
        augmentor_dir = os.path.abspath(os.path.join(self.image_path, os.pardir))
        augmentor_path = os.path.join(augmentor_dir, "augmentor.pkl")
        if os.path.exists(augmentor_path):
            print(f"Loading existing augmentor from {augmentor_path}")
            with open(augmentor_path, "rb") as f:
                self.augmentor = pickle.load(f)
        else:
            print(f"Creating new augmentor for {self.image_path} and {self.colmap_path}")
            self.augmentor = Augmentor(self.image_path, self.colmap_path, mode="full")
            with open(augmentor_path, "wb") as f:
                pickle.dump(self.augmentor, f)
            print(f"Augmentor saved to {augmentor_path}")

    def _compute_image_gradients(self):
        """Computes Sobel gradients for all training images."""
        images = {
            image_id: cv2.cvtColor(cv2.imread(os.path.join(self.augmentor.image_dir, self.augmentor.colmap_images[image_id].name)), cv2.COLOR_BGR2GRAY) / 255.
            for image_id in self.augmentor.image_keys
        }
        self.ixs = {}
        self.iys = {}
        for image_id, image in images.items():
            gauss = cv2.GaussianBlur(image, (5, 5), 0)
            self.ixs[image_id] = cv2.blur(cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=5), (5, 5))
            self.iys[image_id] = cv2.blur(cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=5), (5, 5))

    def gradient_to_covariance(self, image_id, point2D_coord, corresponding_width):
        """Converts image gradients to a 2D covariance matrix."""
        # import pdb; pdb.set_trace()
        point2D_coordx, point2D_coordy = point2D_coord
        ix = self.ixs[image_id][point2D_coordy, point2D_coordx]
        iy = self.iys[image_id][point2D_coordy, point2D_coordx]
        grad_image = np.asarray([[ix, iy]])
        grad_magnitude = np.linalg.norm(grad_image)
        alpha = 1 / (1 + np.exp(-grad_magnitude))
        structure_tensor = np.array([[ix * ix, ix * iy], [ix * iy, iy * iy]])
        eigenvalues, eigenvectors = np.linalg.eigh(structure_tensor)
        s = eigenvalues[0] + 1e-3
        anisotropic_part = np.diag(s/(eigenvalues + 1e-3))
        anisotropic_part[1,1] = np.clip(anisotropic_part[1,1], 0.1, None)
        scale_matrix = alpha * anisotropic_part + (1 - alpha) * np.diag([1., 1.])
        cov_matrix = eigenvectors @ scale_matrix @ eigenvectors.T
        return cov_matrix
    
    def computeWorldToPix(self, point3Dcoord, image_id):
        """Computes the 2D pixel coordinates from a 3D point using the camera's projection matrix."""
        intrinsic_matrix = self.augmentor.intrinsics_camera[1]
        rotation_matrix = self.augmentor.rotations_image[image_id]
        translation_vector = self.augmentor.translations_image[image_id]

        # Convert 3D point to homogeneous coordinates
        X = point3Dcoord.reshape(3, 1)
        
        # Apply camera transformation
        Xh_cam = rotation_matrix @ X + translation_vector.reshape(3, 1)

        # Project to pixel coordinates
        Xh_pix = intrinsic_matrix @ Xh_cam
        x_pix, y_pix = Xh_pix[:2] / Xh_pix[2]
        
        return x_pix, y_pix
    
    def compute_depth(self, point3Dcoord, image_id):
        rotation_matrix = self.augmentor.rotations_image[image_id]
        translation_vector = self.augmentor.translations_image[image_id].reshape(3,1)
        X = point3Dcoord.reshape(3, 1)
        Xcam = rotation_matrix @ X + translation_vector
        depth = Xcam[2, 0]  # Z coordinate in camera space
        return depth
    
    def compute_scales(self, depths):
        depths = np.asarray(depths)
        depth_mean = np.mean(depths)
        return (depth_mean / depths) ** 2


    #################################################
    # This part is for directly predicting 3D covariance matrices from COLMAP 3D points.

    def vech(self, matrix):
        n = matrix.shape[0]
        if matrix.shape[1] == matrix.shape[2] == 2:
            vech_matrix = np.zeros((n, 3))
            vech_matrix[:, 0] = matrix[:, 0, 0]
            vech_matrix[:, 1] = matrix[:, 1, 1]
            vech_matrix[:, 2] = matrix[:, 0, 1]
        return vech_matrix
    
    def kroneckerJacobianMatrices(self, jacobian_matrices):
        n_images = jacobian_matrices.shape[0]
        J = np.zeros((n_images, 3, 6))
        j11 = jacobian_matrices[:, 0, 0]
        j12 = jacobian_matrices[:, 0, 1]
        j13 = jacobian_matrices[:, 0, 2]
        j21 = jacobian_matrices[:, 1, 0]
        j22 = jacobian_matrices[:, 1, 1]
        j23 = jacobian_matrices[:, 1, 2]
        J[:, 0, 0] = j11**2
        J[:, 0, 1] = j12**2
        J[:, 0, 2] = j13**2
        J[:, 0, 3] = 2*j11*j12
        J[:, 0, 4] = 2*j11*j13
        J[:, 0, 5] = 2*j12*j13
        J[:, 1, 0] = j21**2
        J[:, 1, 1] = j22**2
        J[:, 1, 2] = j23**2
        J[:, 1, 3] = 2*j21*j22
        J[:, 1, 4] = 2*j21*j23
        J[:, 1, 5] = 2*j22*j23
        J[:, 2, 0] = j11*j21
        J[:, 2, 1] = j12*j22
        J[:, 2, 2] = j13*j23
        J[:, 2, 3] = j11*j22 + j12*j21
        J[:, 2, 4] = j11*j23 + j13*j21
        J[:, 2, 5] = j12*j23 + j13*j22
        return J.reshape(n_images*3, 6)

    def cholesky_to_sigma3D(self, params):
        l00 = np.exp(params[0])
        l11 = np.exp(params[1])
        l22 = np.exp(params[2])
        l01 = params[3]
        l02 = params[4]
        l12 = params[5]

        L = np.array([[l00, 0, 0],
                      [l01, l11, 0],
                      [l02, l12, l22]])
        return L @ L.T
    
    def sigma3D_to_vech(self, sigma3D):
        return np.array([
            sigma3D[0, 0],  # xx
            sigma3D[1, 1],  # yy
            sigma3D[2, 2],  # zz
            sigma3D[0, 1],  # xy
            sigma3D[0, 2],  # xz
            sigma3D[1, 2]   # yz
        ])
    
    def loss_fn_cholesky(self, params, J, cov2D_vech):
        Sigma3D = self.cholesky_to_sigma3D(params)
        n = J.shape[0] // 3
        proj_cov2D_vech = np.zeros((n, 3))
        for i in range(n):
            Ji = J[3*i:3*(i+1)]
            x = self.sigma3D_to_vech(Sigma3D)
            proj_cov2D_vech[i] = Ji @ x

        residual = proj_cov2D_vech.reshape(-1) - cov2D_vech.reshape(-1)
        return np.sum(residual**2)
    
    def computeJacobianMatrices(self, point3Dcoord, imageids, Ks, Rs, ts):
        """
        Computes the Jacobian matrices for the 3D points based on the provided parameters.
        """
        assert len(imageids) == Ks.shape[0] == Rs.shape[0] == ts.shape[0], "Batch dimensions must match"
        n_images = len(imageids)
        X_world = point3Dcoord.reshape(-1, 3, 1)
        X_cam = np.matmul(Rs, X_world) + ts
        X_cam = X_cam.reshape(n_images, 3)
        J = np.zeros((n_images, 2, 3))  # Jacobian for each image
        J[:, 0, 0] = Ks[:, 0, 0] / X_cam[:, 2]
        J[:, 1, 1] = Ks[:, 1, 1] / X_cam[:, 2]
        J[:, 0, 2] = -Ks[:, 0, 0] * X_cam[:, 0] / (X_cam[:, 2] ** 2)
        J[:, 1, 2] = -Ks[:, 1, 1] * X_cam[:, 1] / (X_cam[:, 2] ** 2)
        J = np.matmul(J, Rs)
        # Returns (n_images, 2, 3) Jacobian matrices
        return J

    def solveCholeskyMinimizeCov2D3D(self, **kwargs):
        assert "point3Dcoord" in kwargs, "point3Dcoord must be provided"
        assert "image_ids" in kwargs, "image_ids must be provided"
        assert "cov2Dmatrix" in kwargs, "cov2Dmatrix must be provided"
        assert len(kwargs["image_ids"]) == len(kwargs["cov2Dmatrix"]), "image_ids and cov2Dmatrix must have the same length"
        assert len(kwargs["image_ids"]) > 0, "At least one image ID must be provided"

        from scipy.optimize import minimize
        jacobian_matrices = self.computeJacobianMatrices(kwargs["point3Dcoord"],
                                                            kwargs["image_ids"],
                                                         np.asarray([self.augmentor.intrinsics_camera[self.augmentor.colmap_images[image_id].camera_id] for image_id in kwargs["image_ids"]]),
                                                         np.asarray([self.augmentor.rotations_image[image_id] for image_id in kwargs["image_ids"]]),
                                                         np.asarray([self.augmentor.translations_image[image_id].reshape(3, 1) for image_id in kwargs["image_ids"]]))
        assert jacobian_matrices.shape[0] == len(kwargs["image_ids"])
        assert jacobian_matrices.shape[1] == 2
        assert jacobian_matrices.shape[2] == 3

        if len(kwargs["cov2Dmatrix"]) > 1:
            cov2Dmatrices = np.asarray([
                kwargs["cov2Dmatrix"]
            ]).reshape((len(kwargs["image_ids"]), 2, 2))
        else:
            return None

        vech_c2D = self.vech(cov2Dmatrices)
        J = self.kroneckerJacobianMatrices(jacobian_matrices)

        init_params = np.array([0., 0., 0., 0., 0., 0.])
        result = minimize(self.loss_fn_cholesky, init_params, args=(J, vech_c2D), method='L-BFGS-B')

        Sigma3D_est = self.cholesky_to_sigma3D(result.x)
        return Sigma3D_est
    
    def computeJacobian(self, camcoords, K, R):
        assert camcoords.shape[0] == K.shape[0] == R.shape[0], "Batch dimensions must match"
        B = camcoords.shape[0]
        camcoords = camcoords.reshape(B, 3)  # Ensure camcoords is (B, 3)
        K = K.reshape(B, 3, 3)  # Ensure K is (B, 3, 3)
        R = R.reshape(B, 3, 3)  # Ensure R is (B, 3, 3)

        J = np.zeros((B, 2, 3))
        J[:, 0, 0] = K[:, 0, 0] / camcoords[:, 2]
        J[:, 1, 1] = K[:, 1, 1] / camcoords[:, 2]
        J[:, 0, 2] = -J[:, 0, 0] * camcoords[:, 0] / camcoords[:, 2]
        J[:, 1, 2] = -J[:, 1, 1] * camcoords[:, 1] / camcoords[:, 2]
        J = np.einsum('bij,bjk->bik', J, R)  # Apply rotation
        return J

    def project3D2D(self, point3Dcoord, image_ids, sigma3D=None):
        if len(image_ids) == 0:
            return None
        
        intrinsic_matrices = np.stack([self.augmentor.intrinsics_camera[self.augmentor.colmap_images[image_id].camera_id] for image_id in image_ids])
        rotation_matrices = np.stack([self.augmentor.rotations_image[image_id] for image_id in image_ids])
        translation_vectors = np.stack([self.augmentor.translations_image[image_id].reshape(3,1) for image_id in image_ids])
        point3D_world = point3Dcoord.reshape(3, 1)
        point3D_camera = np.einsum('bij,jk->bik', rotation_matrices, point3D_world) + translation_vectors
        point3D_pixel = np.einsum('bij,bjk->bik', intrinsic_matrices, point3D_camera)
        point3D_uv, point3D_depth = point3D_pixel[:, :2, :] / point3D_pixel[:, 2:, :], point3D_pixel[:, 2:, :] # (B, 2, 1), (B, 1, 1)
        J = self.computeJacobian(point3D_camera, intrinsic_matrices, rotation_matrices)
        if sigma3D is not None:
            cov3Dmatrix_proj = np.einsum('bij,jk,bkl->bil', J, sigma3D, J.transpose(0, 2, 1))
        else:
            cov3Dmatrix_proj = None
        point3D_uv = np.clip(np.round(point3D_uv.reshape(-1, 2)), 0, intrinsic_matrices[:, :2, 2].reshape(-1, 2)*2-1).astype(int)
        return {"point3D_uv": point3D_uv, "cov3Dmatrix_proj": cov3Dmatrix_proj, "sigma3D": sigma3D, "point3D_depth": point3D_depth}

    def estimateCov3D(self, point3Dcoord, image_ids, patch_length=0.05):
        # project 3D point to 2D for constructing 2D covariance matrix labels
        proj = self.project3D2D(point3Dcoord, image_ids)
        if proj is None:
            return None
        uvs = proj["point3D_uv"]
        depths = proj["point3D_depth"].reshape(-1)
        focal_lengths = np.asarray([self.augmentor.intrinsics_camera[self.augmentor.colmap_images[image_id].camera_id][0:2, 2] for image_id in image_ids])
        focal_length_mean = np.mean(focal_lengths)
        # Ensure focal_length_mean is a scalar
        if hasattr(focal_length_mean, 'item'):
            focal_length_mean = focal_length_mean.item()
        assert np.isscalar(focal_length_mean), "focal_length_mean must be a scalar"
        window_sizes = np.floor(focal_length_mean * patch_length / depths)
        window_sizes = np.maximum(window_sizes, 3)
        window_sizes = (window_sizes // 2) * 2 + 1
        window_sizes = window_sizes.astype(int)
        Ix_means = [cv2.GaussianBlur(self.Ixs[image_id], (int(window_size), int(window_size)), 0)[uvs[i][1], uvs[i][0]] for i, (image_id, window_size) in enumerate(zip(image_ids, window_sizes))]
        Iy_means = [cv2.GaussianBlur(self.Iys[image_id], (int(window_size), int(window_size)), 0)[uvs[i][1], uvs[i][0]] for i, (image_id, window_size) in enumerate(zip(image_ids, window_sizes))]
        structure_tensors = [np.asarray([[Ix**2, Ix*Iy], [Ix*Iy, Iy**2]]) for Ix, Iy in zip(Ix_means, Iy_means)]
        eigdecs = [np.linalg.eigh(st.reshape(2,2)) for st in structure_tensors]
        eigvals = [eigdec[0] for eigdec in eigdecs]
        eigvecs = [eigdec[1] for eigdec in eigdecs]
        adapatch_cov2Ds = []
        for i in range(len(eigvals)):
            grad_magnitude = np.linalg.norm(np.asarray([Ix_means[i], Iy_means[i]]))
            alpha = 1 / (1 + np.exp(-grad_magnitude))

            rel_eigvals = eigvals[i] + 1e-6
            rel_scale = (np.min(rel_eigvals) / rel_eigvals)
            rel_scale = np.clip(rel_scale, 0.1, 1)
            max_scale = (window_sizes[i] / 2) ** 2
            aniso_part = np.diag(max_scale * rel_scale)
            iso_part = np.diag([max_scale, max_scale])
            scale_matrix = alpha * aniso_part + (1 - alpha) * iso_part
            eigvals_tmp, _ = np.linalg.eigh(scale_matrix)
            k = max_scale / np.max(eigvals_tmp)
            scale_matrix = scale_matrix * k
            cov2D = eigvecs[i] @ scale_matrix @ eigvecs[i].T
            adapatch_cov2Ds.append(cov2D)
        adapatch_cov2Ds = np.asarray(adapatch_cov2Ds)

        Sigma3D_est = self.solveCholeskyMinimizeCov2D3D(
            point3Dcoord=point3Dcoord,
            image_ids=image_ids,
            cov2Dmatrix=adapatch_cov2Ds
        )

        eigval, eigvec = np.linalg.eigh(Sigma3D_est.reshape(3, 3))
        proj_temp = self.project3D2D(point3Dcoord, image_ids, Sigma3D_est.reshape(3,3))
        sfs = []
        for i, cov2d_temp in enumerate(proj_temp["cov3Dmatrix_proj"]):
            assert cov2d_temp is not None
            eigval_temp, eigvec_temp = np.linalg.eigh(cov2d_temp)
            target_size = window_sizes[i] / 2
            sf = target_size / np.sqrt(eigval_temp[1])
            sfs.append(sf)
        scale_factor = np.mean(np.asarray(sfs))
        Sigma3D_est = eigvec @ np.diag(eigval * scale_factor**2) @ eigvec.T
        
        return Sigma3D_est
    
    def rotation_to_quaternion(self, rotation_matrix):
        assert rotation_matrix.shape == (3, 3), "Rotation matrix must be 3x3"
        r = rotation_matrix
        qw = torch.sqrt(torch.clamp(1 + r[0, 0] + r[1, 1] + r[2, 2], min=0)) / 2
        qx = torch.sqrt(torch.clamp(1 + r[0, 0] - r[1, 1] - r[2, 2], min=0)) / 2
        qy = torch.sqrt(torch.clamp(1 - r[0, 0] + r[1, 1] - r[2, 2], min=0)) / 2
        qz = torch.sqrt(torch.clamp(1 - r[0, 0] - r[1, 1] + r[2, 2], min=0)) / 2
        qx = torch.copysign(qx, r[2, 1] - r[1, 2])
        qy = torch.copysign(qy, r[0, 2] - r[2, 0])
        qz = torch.copysign(qz, r[1, 0] - r[0, 1])
        return torch.tensor([qw, qx, qy, qz], dtype=torch.float32, device="cuda")

    def checkBaseline(self, point3Dcoord, image_ids):
        Rs = np.asarray([self.augmentor.rotations_image[image_id] for image_id in image_ids])
        ts = np.asarray([self.augmentor.translations_image[image_id].reshape(3,1) for image_id in image_ids])
        assert Rs.shape == (len(image_ids), 3, 3)
        assert ts.shape == (len(image_ids), 3, 1)
        camcenters = np.asarray([(-R.T @ t).reshape(3) for R, t in zip(Rs, ts)])
        assert camcenters.shape == (len(image_ids), 3)
        point3Dcoord = point3Dcoord.reshape(3)
        # Compute all angles between the vector pairs from camera centers to the 3D point
        vectors = camcenters - point3Dcoord
        assert vectors.shape == (len(image_ids), 3)
        norms = np.linalg.norm(vectors, axis=1)
        assert norms.shape == (len(image_ids),)
        vectors_normalized = vectors / norms[:, np.newaxis]
        assert vectors_normalized.shape == (len(image_ids), 3)
        angles = np.arccos(np.clip(np.dot(vectors_normalized, vectors_normalized.T), -1.0, 1.0))
        assert angles.shape == (len(image_ids), len(image_ids))
        # Check if the minimum angle is less than 10 degrees
        max_angle = np.max(angles)  # Exclude zero
        if max_angle < np.deg2rad(10):
            return "narrow"
        else:
            return "wide"

    def directApplicationCov3Ds(self, GaussianModel):
        self.images = {image_id: cv2.cvtColor(cv2.imread(os.path.join(self.augmentor.image_dir, self.augmentor.colmap_images[image_id].name)), cv2.COLOR_BGR2GRAY)/255. for image_id in self.augmentor.image_keys}
        self.Ixs = {image_id: cv2.Sobel(cv2.GaussianBlur(self.images[image_id], (5,5), 0), cv2.CV_64F, 1, 0, ksize=5) for image_id in self.augmentor.image_keys}
        self.Iys = {image_id: cv2.Sobel(cv2.GaussianBlur(self.images[image_id], (5,5), 0), cv2.CV_64F, 0, 1, ksize=5) for image_id in self.augmentor.image_keys}
        c = 0
        Sigma3Ds = {}
        for point3D_idx, (_, point3D) in tqdm(enumerate(self.augmentor.colmap_points3D.items()), desc="Estimating 3D Covariance Matrices"):
            image_ids = [point3D.image_ids[i] for i in range(len(point3D.image_ids)) if point3D.image_ids[i] in self.augmentor.image_keys]
            if len(image_ids) < 2:
                continue
            point3Dcoord = np.asarray(point3D.xyz, dtype=np.float32)
            if self.checkBaseline(point3Dcoord, image_ids) == "narrow":
                continue
            Sigma3D_est = self.estimateCov3D(point3Dcoord, image_ids)
            if Sigma3D_est is not None:
                Sigma3Ds[point3D_idx] = Sigma3D_est
                c += 1
        # Decompose each 3D covariance matrix into rotation (R) and log-scale (S) matrices such that Sigma = R @ exp(S) @ exp(S) @ R.T
        self.cov3d_decomposed = {}
        for idx, Sigma in Sigma3Ds.items():
            # Eigen-decomposition: Sigma = V @ diag(eigvals) @ V.T
            eigvals, eigvecs = np.linalg.eigh(Sigma)
            # Ensure positive eigenvalues for log
            eigvals = np.clip(eigvals, 1e-8, None)
            # S is diagonal matrix of log(sqrt(eigvals))
            S = np.log(np.sqrt(eigvals))
            R = eigvecs
            self.cov3d_decomposed[idx] = {"R": R, "S": S}
        with torch.no_grad():
            for point3D_idx, cov3d in self.cov3d_decomposed.items():
                R = torch.tensor(cov3d["R"], dtype=torch.float32, device="cuda")
                S = torch.tensor(cov3d["S"], dtype=torch.float32, device="cuda")
                GaussianModel._rotation[point3D_idx] = torch.tensor(self.rotation_to_quaternion(R), dtype=torch.float32, device="cuda")
                GaussianModel._scaling[point3D_idx] = torch.tensor(S, dtype=torch.float32, device="cuda")

        print(f"Estimated {c} 3D covariance matrices from COLMAP points.")
            

    #################################################


    



    def create_p3d_image_id_matrix(self):
        from points3Dvisualization import Colmap3DPoint
        """Creates a matrix mapping 3D points to image IDs and their 2D coordinates."""
        self.p3d_image_id_matrix = []
        self.corresponding_cov_matrix = []
        self.test_only_3d_indices = []
        self.colmap3Dpoints = {}
        self.images = {image_id: cv2.cvtColor(cv2.imread(os.path.join(self.augmentor.image_dir, self.augmentor.colmap_images[image_id].name)), cv2.COLOR_BGR2GRAY)/255. for image_id in self.augmentor.image_keys}
        self.Ixs = {image_id: cv2.Sobel(cv2.GaussianBlur(self.images[image_id], (5,5), 0), cv2.CV_64F, 1, 0, ksize=5) for image_id in self.augmentor.image_keys}
        self.Iys = {image_id: cv2.Sobel(cv2.GaussianBlur(self.images[image_id], (5,5), 0), cv2.CV_64F, 0, 1, ksize=5) for image_id in self.augmentor.image_keys}


        for point3D_idx, (_, point3D) in tqdm(enumerate(self.augmentor.colmap_points3D.items()), desc="Creating Point-Image Matrix"):
            imagepath = self.augmentor.image_dir
            imageids = []
            imagenames = []
            point2Dcoords = []
            cov2Dmatrices = []
            point3Dcoord = point3D.xyz
            intrinsic_matrices = self.augmentor.intrinsics_camera
            rotation_matrices = self.augmentor.rotations_image
            translation_vectors = self.augmentor.translations_image
            depths = []
            for i, image_id in enumerate(point3D.image_ids):
                if image_id not in self.augmentor.image_keys:
                    continue
                depth = self.compute_depth(point3Dcoord, image_id)
                depths.append(depth)

            scales = self.compute_scales(depths)
            idx = 0
            for i, image_id in enumerate(point3D.image_ids):
                if image_id not in self.augmentor.image_keys:
                    # self.test_only_3d_indices.append(point3D_idx)
                    continue
                
                point2D_idx = point3D.point2D_idxs[i]
                # point2D_coordx, point2D_coordy = self.augmentor.colmap_images[image_id].xys[point2D_idx]
                point2D_coordx, point2D_coordy = self.computeWorldToPix(point3Dcoord, image_id)
                
                # Modified version has intrinsic matrices that are already scaled, so there is no need for scaling.
                # scaled_x = np.clip(round(point2D_coordx / self.image_scale), 0, self.augmentor.roots[image_id].width - 1)
                # scaled_y = np.clip(round(point2D_coordy / self.image_scale), 0, self.augmentor.roots[image_id].height - 1)
                
                # corresponding_leaf_node = self.augmentor.find_corresponding_leaf_node(image_id, [scaled_x, scaled_y])
                # if corresponding_leaf_node is None:
                #     continue

                scaled_x = int(np.clip(np.round(point2D_coordx), 0, self.augmentor.roots[image_id].width - 1))
                scaled_y = int(np.clip(np.round(point2D_coordy), 0, self.augmentor.roots[image_id].height - 1))
                cov_matrix = self.gradient_to_covariance(image_id, [scaled_x, scaled_y], None)
                
                if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
                    continue
                    
                self.p3d_image_id_matrix.append([point3D_idx, image_id, None, None])
                self.corresponding_cov_matrix.append(cov_matrix * scales[idx])
                imageids.append(image_id)
                imagenames.append(self.augmentor.colmap_images[image_id].name)
                point2Dcoords.append([scaled_x, scaled_y])
                cov2Dmatrices.append(cov_matrix * scales[idx])
                idx += 1
            
            self.colmap3Dpoints[point3D_idx] = Colmap3DPoint(
                imageids=imageids,
                imagenames=imagenames,
                imagepath=imagepath,
                point2Dcoords=point2Dcoords,
                cov2Dmatrices=cov2Dmatrices,
                point3Dcoord=point3Dcoord,
                cov3Dmatrix=None,
                intrinsic_matrices=intrinsic_matrices,
                rotation_matrices=rotation_matrices,
                translation_vectors=translation_vectors,
            )



    
    def augmentation_mode(self):
        """Switches the dataset to augmentation mode."""
        self.mode = "augmentation"
        self.p3d_ids = [id + self.n_colmap_points for id in self.augmentor.merged_sample_points_indices]
        self.p3d_image_ids = self.augmentor.merged_image_ids
        self.p3d_covs = torch.tensor(np.asarray(self.augmentor.cov_labels, dtype=np.float32) * self.scale, device="cuda")

    def __len__(self):
        return len(self.p3d_ids) if self.mode == "augmentation" else len(self.p3d_image_id_matrix)

    def __getitem__(self, idx):
        if self.mode == "augmentation":
            return (self.p3d_ids[idx], self.p3d_image_ids[idx]), self.p3d_covs[idx]
        else:
            return (self.p3d_image_id_matrix[idx][0], self.p3d_image_id_matrix[idx][1]), self.corresponding_cov_matrix[idx]
