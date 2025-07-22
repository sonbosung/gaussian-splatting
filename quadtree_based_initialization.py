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

class GaussianInitializer:
    """
    Initializes Gaussian attributes (features, scaling, rotation) based on 2D covariance
    information derived from a quadtree decomposition of the input images.
    """
    def __init__(self, gaussians, cameras, dataset, n_steps=100000, lr=1e-2, batch_size=256, pipe=None, background=None, gs_dataset=None, opt=None, args=None):
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
            {'params': [self.gaussians._features_dc], 'lr': self.lr*0.1, "name": "features_dc"},
            {'params': [self.gaussians._scaling], 'lr': self.lr*2.0, "name": "scaling"},
            {'params': [self.gaussians._rotation], 'lr': self.lr*0.1, "name": "rotation"}
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
        kl_loss = self.kl_divergence_2d_gaussian(None, gt_cov, None, pred_cov2d)

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
        return kl_loss


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

    def camera_ids_to_indices(self):
        """Maps COLMAP image IDs to camera indices used in the training scene."""
        colmap_image_id_names = {id: colmap_image.name for id, colmap_image in self.dataset.augmentor.colmap_images.items()}
        camera_name_index = {camera.image_name: idx for idx, camera in enumerate(self.cameras)}
        
        self.colmap_image_id_to_camera_index = {}
        for id, name in colmap_image_id_names.items():
            if name in camera_name_index:
                self.colmap_image_id_to_camera_index[id] = camera_name_index[name]

        valid_entries = []
        for entry in self.dataset.p3d_image_id_matrix:
            colmap_image_id = entry[1]
            if colmap_image_id in self.colmap_image_id_to_camera_index:
                entry[1] = self.colmap_image_id_to_camera_index[colmap_image_id]
                valid_entries.append(entry)
        
        print(f"Filtered {len(self.dataset.p3d_image_id_matrix) - len(valid_entries)} entries corresponding to test cameras.")
        self.dataset.p3d_image_id_matrix = valid_entries

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
        
        self._compute_image_gradients()
        self.create_p3d_image_id_matrix()
        self.mode = "sfm"
        self.n_colmap_points = len(self.augmentor.colmap_points3D)

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
            self.ixs[image_id] = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=5)
            self.iys[image_id] = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=5)

    def gradient_to_covariance(self, image_id, point2D_coord, corresponding_width):
        """Converts image gradients to a 2D covariance matrix."""
        point2D_coordx, point2D_coordy = point2D_coord
        ix = self.ixs[image_id][point2D_coordy, point2D_coordx]
        iy = self.iys[image_id][point2D_coordy, point2D_coordx]
        
        # if abs(ix) < 1e-3 and abs(iy) < 1e-3:
        #     cov_matrix = np.asarray([[corresponding_width**2 / 36, 0], [0, corresponding_width**2 / 36]])
        # elif abs(ix) < 1e-3:
        #     cov_matrix = np.asarray([[corresponding_width**2 / (36 * 9), 0], [0, corresponding_width**2 / 36]])
        # elif abs(iy) < 1e-3:
        #     cov_matrix = np.asarray([[corresponding_width**2 / 36, 0], [0, corresponding_width**2 / (36 * 9)]])
        # else:
        #     structure_tensor = np.array([[ix**2, ix * iy], [ix * iy, iy**2]]) + 1e-6
        #     eigvals, eigvecs = np.linalg.eigh(structure_tensor)
        #     eigvals_max = np.max(eigvals)
        #     eigvals = np.maximum(eigvals / eigvals_max, 0.2)
        #     eigvals = np.clip(eigvals, 0.2, 1.0)
        #     eigvals = np.sqrt(eigvals) * corresponding_width / 6
        #     scale_matrix = np.diag([eigvals[1], eigvals[0]])
        #     cov_matrix = eigvecs @ scale_matrix @ scale_matrix @ eigvecs.T
        structure_tensor = np.array([[ix**2, ix * iy], [ix * iy, iy**2]])
        eigvals, eigvecs = np.linalg.eigh(structure_tensor)
        eigvals = np.maximum(eigvals, 1e-6) # Ensures no zero eigenvalues
        # Eigvals are sorted in ascending order
        s = np.sqrt(eigvals[0])
        sigma = np.diag(np.maximum(s / np.sqrt(eigvals), 0.33))
        cov_matrix = eigvecs @ sigma @ sigma @ eigvecs.T
        return cov_matrix

    def create_p3d_image_id_matrix(self):
        from points3Dvisualization import Colmap3DPoint
        """Creates a matrix mapping 3D points to image IDs and their 2D coordinates."""
        self.p3d_image_id_matrix = []
        self.corresponding_cov_matrix = []
        self.test_only_3d_indices = []
        self.colmap3Dpoints = {}
        
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

            for i, image_id in enumerate(point3D.image_ids):
                if image_id not in self.augmentor.image_keys:
                    self.test_only_3d_indices.append(point3D_idx)
                    continue
                
                point2D_idx = point3D.point2D_idxs[i]
                point2D_coordx, point2D_coordy = self.augmentor.colmap_images[image_id].xys[point2D_idx]
                
                scaled_x = np.clip(round(point2D_coordx / self.image_scale), 0, self.augmentor.roots[image_id].width - 1)
                scaled_y = np.clip(round(point2D_coordy / self.image_scale), 0, self.augmentor.roots[image_id].height - 1)
                
                corresponding_leaf_node = self.augmentor.find_corresponding_leaf_node(image_id, [scaled_x, scaled_y])
                if corresponding_leaf_node is None:
                    continue
                    
                cov_matrix = self.gradient_to_covariance(image_id, [scaled_x, scaled_y], corresponding_leaf_node.width)
                
                if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
                    continue
                    
                self.p3d_image_id_matrix.append([point3D_idx, image_id, None, None])
                self.corresponding_cov_matrix.append(cov_matrix * self.scale)
                imageids.append(image_id)
                imagenames.append(self.augmentor.colmap_images[image_id].name)
                point2Dcoords.append([scaled_x, scaled_y])
                cov2Dmatrices.append(cov_matrix * self.scale)
            
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
