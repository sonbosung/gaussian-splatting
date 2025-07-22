import torch
import numpy as np
import os, sys
from colmap.scripts.python.read_write_model import *
from augmentation_new import Augmentor
from torch.utils.data import Dataset
from utils.graphics_utils import BasicPointCloud
from scene.gaussian_model import GaussianModel
from utils.general_utils import build_rotation, build_scaling_rotation
from tqdm import tqdm
import pdb
from utils.sh_utils import SH2RGB
from gaussian_renderer import render, network_gui
import cv2

import torch
import torch.nn as nn
import numpy as np

class GaussianInitializer:
    def __init__(self, gaussians, cameras, dataset, n_steps=50000, lr=1e-2, batch_size=256, pipe=None, background=None, gs_dataset=None, opt=None, args=None):
        self.gaussians = gaussians  # 이미 생성된 모델 참조
        self.cameras = cameras  # 카메라 정보
        self.dataset = dataset
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.world_view_transform = torch.stack([camera.world_view_transform.transpose(0,1) for camera in cameras])
        self.proj_matrix = torch.stack([camera.projection_matrix.transpose(0,1) for camera in cameras])
        self.image_width_height = torch.stack([torch.tensor([camera.image_width, camera.image_height], dtype=torch.float32, device="cuda") for camera in cameras])
        print(len(dataset.p3d_image_id_matrix), "points in dataset")
        self.camera_ids_to_indices()
        self.dataset.corresponding_cov_matrix = torch.tensor(np.stack(self.dataset.corresponding_cov_matrix), dtype=torch.float32, device="cuda")
        # self.dataset.corresponding_rgb_values = torch.tensor(np.stack(self.dataset.corresponding_rgb_values), dtype=torch.float32, device="cuda")
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.args = args
        self.pipe = pipe
        self.background = background
        self.iteration = 0
        self.gs_dataset = gs_dataset
        self.opt = opt
        self.intrinsics_scaling_factor = int(self.dataset.image_path.split("/")[-1].split("_")[-1])**2 if "images_" in self.dataset.image_path else 1
        self.mode = "sfm"  # 초기 모드는 sfm, augmentation 모드로 전환 가능

    def run(self):
        assert self.mode is not "augmentation", "run() should not be called in augmentation mode"
        params = [self.gaussians._features_dc, self.gaussians._scaling, self.gaussians._rotation]
        for p, name in zip(params, ["features_dc", "scaling", "rotation"]):
            p = 1
        optimizer = torch.optim.Adam(params, lr=self.lr)
        dataloader_iter = iter(self.dataloader)
        progress_bar = tqdm(range(self.n_steps), desc="Training progress")
        for step in progress_bar:
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifier=scaling_modifer, use_trained_exp=self.gs_dataset.train_test_exp, separate_sh=False)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, self.gs_dataset.source_path)
                    if do_training and ((self.iteration < int(self.opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None
            try:
                batch_x, batch_y = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                batch_x, batch_y = next(dataloader_iter)

            batch_point_idx, batch_camera_id = batch_x # (B, ), (B, )
            batch_gt_cov = batch_y # (B, 2, 2), (B, 3)
            # batch_gt_cov, batch_gt_rgb = batch_y # (B, 2, 2), (B, 3)
            if torch.isnan(batch_gt_cov).any() or torch.isinf(batch_gt_cov).any():
                print("NaN 또는 Inf가 batch_gt_cov에 포함되어 있습니다.")
                nan_inf_mask = torch.isnan(batch_gt_cov).any(dim=(1,2)) | torch.isinf(batch_gt_cov).any(dim=(1,2))
                print("NaN/Inf indices:", torch.nonzero(nan_inf_mask).squeeze(-1).tolist())
                print("batch_point_idx (problematic):", batch_point_idx[nan_inf_mask])
                print("batch_camera_id (problematic):", batch_camera_id[nan_inf_mask])
                exit(1)
                continue

            batch_pred_xyz = self.gaussians._xyz[batch_point_idx] # (B, 3)
            # print("batch_point_idx: ", batch_point_idx)
            # print("features_dc shape: ", self.gaussians._features_dc.shape)
            batch_pred_rgb = SH2RGB(self.gaussians._features_dc[batch_point_idx]) # (B, 1, 3)
            batch_Rt = self.world_view_transform[batch_camera_id]  # (B, 4, 4)
            batch_proj_matrix = self.proj_matrix[batch_camera_id]  # (B, 4, 4)
            batch_image_width = self.image_width_height[batch_camera_id][:, 0]  # (B, )
            batch_image_height = self.image_width_height[batch_camera_id][:, 1]  # (B, )
            batch_intrinsic_matrix = self.P_gl_to_P_cv(batch_proj_matrix, batch_image_width, batch_image_height)
            assert batch_intrinsic_matrix.shape == (len(batch_point_idx), 3, 3)
            batch_pmat = torch.bmm(batch_intrinsic_matrix, batch_Rt[:, :3, :4])  # (B, 3, 4)
            
            # batch_pred_scale = torch.exp(self.gaussians._scaling[batch_point_idx])  # (B, 3)
            # batch_pred_rot = self.gaussians._rotation[batch_point_idx]  # (B, 4)
            # pdb.set_trace()
            batch_pred_cov = self.gaussians.get_covariance()[batch_point_idx]  # (B, 6)
            batch_pred_cov3d = self.uppertri_to_symm(batch_pred_cov)
            # batch_pred_cov3d = self.compute_3d_covariance_batch(batch_pred_scale, batch_pred_rot)
            batch_pred_cov2d = self.project_cov3d_to_2d_batch(batch_pred_cov3d, batch_pmat, batch_pred_xyz)
            batch_pred_cov2d = batch_pred_cov2d + torch.eye(batch_pred_cov2d.size(-1), device=batch_pred_cov2d.device) * 1e-6
            # loss_cov = self.kl_divergence_2d_gaussian_batch(batch_gt_cov, batch_pred_cov2d).mean()
            loss_cov = self.frobenius_norm_batch(batch_gt_cov, batch_pred_cov2d).mean()
            loss = loss_cov
            # loss_rgb = ((batch_pred_rgb - batch_gt_rgb.unsqueeze(1)/255.) ** 2).mean()
            # loss = loss_cov + loss_rgb

            if torch.isnan(loss):
                print(f"NaN detected at step {step}")
                # print("batch_pred_scale:", batch_pred_scale)
                # print("batch_pred_rot:", torch.linalg.norm(batch_pred_rot, dim=0))
                print("batch_point_idx: ", batch_point_idx)
                print("batch_camera_id:", batch_camera_id)
                print("batch_pred_cov2d:", batch_pred_cov2d)
                print("batch_gt_cov:", batch_gt_cov)
                print("batch_pred_rgb:", batch_pred_rgb)
                # print("batch_gt_rgb:", batch_gt_rgb)
                return

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                progress_bar.set_postfix({f"Step": f"{step}", "Loss": f"{loss:.6f}"})

    def run_for_augmentation(self):
        assert self.mode is "augmentation", "run_for_augmentation() should be called in augmentation mode"
        self.n_steps = 50000
        params = [self.gaussians._features_dc, self.gaussians._scaling, self.gaussians._rotation, self.gaussians._xyz]
        for p, name in zip(params, ["features_dc", "scaling", "rotation", "xyz"]):
            p = 1
        optimizer = torch.optim.Adam(params, lr=self.lr)
        self.dataset.p3d_image_ids = [self.colmap_image_id_to_camera_index[image_id] for image_id in self.dataset.p3d_image_ids]
        dataloader_iter = iter(self.dataloader)
        progress_bar = tqdm(range(self.n_steps), desc="Training progress")
        for step in progress_bar:
            try:
                batch_x, batch_y = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                batch_x, batch_y = next(dataloader_iter)
        
            batch_point_idx, batch_camera_id = batch_x # (B, ), (B, )
            batch_gt_cov = batch_y # (B, 2, 2)
            if torch.isnan(batch_gt_cov).any() or torch.isinf(batch_gt_cov).any():
                print("NaN 또는 Inf가 batch_gt_cov에 포함되어 있습니다.")
                continue
            # batch_gt_mean, batch_gt_cov, batch_gt_rgb = batch_y # (B, 2), (B, 2, 2), (B, 3)

            # batch_gt_mean = batch_gt_mean.to("cuda")
            # batch_gt_cov = batch_gt_cov.to("cuda")
            # batch_gt_rgb = batch_gt_rgb.to("cuda")

            batch_pred_xyz = self.gaussians._xyz[batch_point_idx]  # (B, 3)
            batch_pred_rgb = SH2RGB(self.gaussians._features_dc[batch_point_idx])  # (B, 1, 3)
            batch_Rt = self.world_view_transform[batch_camera_id]  # (B, 4, 4)
            batch_proj_matrix = self.proj_matrix[batch_camera_id]  # (B, 4, 4)
            batch_image_width = self.image_width_height[batch_camera_id][:, 0]  # (B, )
            batch_image_height = self.image_width_height[batch_camera_id][:, 1]  # (B, )
            batch_intrinsic_matrix = self.P_gl_to_P_cv(batch_proj_matrix, batch_image_width, batch_image_height)
            assert batch_intrinsic_matrix.shape == (len(batch_point_idx), 3, 3)
            batch_pmat = torch.bmm(batch_intrinsic_matrix, batch_Rt[:, :3, :4])  # (B, 3, 4)
            # pdb.set_trace()

            # batch_pred_mean = self.project_xyz_to_mean_batch(batch_pred_xyz, batch_pmat)  # (B, 2)
            # batch_pred_cov = self.gaussians.get_covariance()[batch_point_idx]  # (B, 6)
            batch_pred_cov = self.gaussians.get_covariance_slice(batch_point_idx)  # (B, 6)
            batch_pred_cov3d = self.uppertri_to_symm(batch_pred_cov)
            batch_pred_cov2d = self.project_cov3d_to_2d_batch(batch_pred_cov3d, batch_pmat, batch_pred_xyz)
            # loss_mean = ((batch_gt_mean - batch_pred_mean) ** 2).mean()
            # loss_cov = self.kl_divergence_2d_gaussian_batch(batch_gt_cov, batch_pred_cov2d).mean()
            loss_cov = self.frobenius_norm_batch(batch_gt_cov, batch_pred_cov2d).mean()
            # loss_rgb = ((batch_pred_rgb - batch_gt_rgb.unsqueeze(1) / 255.) ** 2).mean()
            # loss = loss_mean + loss_cov + loss_rgb
            loss = loss_cov

            # pdb.set_trace()

            if torch.isnan(loss):
                print(f"NaN detected at step {step}")
                print("batch_pred_cov2d:", batch_pred_cov2d)
                print("batch_gt_mean:", batch_gt_mean)
                print("batch_pred_mean:", batch_pred_mean)
                print("batch_gt_cov:", batch_gt_cov)
                print("batch_pred_rgb:", batch_pred_rgb)
                print("batch_gt_rgb:", batch_gt_rgb)
                return
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                # progress_bar.set_postfix({"Loss": f"{loss:.6f}", "mean_loss": f"{loss_mean:.6f}", "cov_loss": f"{loss_cov:.6f}", "rgb_loss": f"{loss_rgb:.6f}"})
                progress_bar.set_postfix({"Loss": f"{loss:.6f}"})

    def project_xyz_to_mean_batch(self, xyz_batch, pmat_batch):
        """
        xyz_batch: (B, 3) torch.Tensor, 배치 단위 3D points
        pmat_batch: (B, 3, 4) torch.Tensor, 배치 단위 projection matrix
        반환값: (B, 2) torch.Tensor, 배치 단위 2D mean points
        """
        B = xyz_batch.shape[0]
        ones = torch.ones((B, 1), dtype=xyz_batch.dtype, device=xyz_batch.device)
        Xh = torch.cat([xyz_batch, ones], dim=1)
        # 2D 좌표 계산
        u = torch.sum(pmat_batch[:, 0] * Xh, dim=1)
        v = torch.sum(pmat_batch[:, 1] * Xh, dim=1)
        w = torch.sum(pmat_batch[:, 2] * Xh, dim=1)
        return torch.stack([u / w, v / w], dim=1)  # (B, 2)

    def augmentation_mode(self):
        """
        Augmentation mode로 설정
        :return: None
        """
        self.mode = "augmentation"
        self.dataset.augmentation_mode()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def P_gl_to_P_cv(self, P_gl, width, height):
        assert P_gl.shape[1:] == (4,4)
        assert width.shape[0] == P_gl.shape[0]
        assert height.shape[0] == P_gl.shape[0]
        B = P_gl.shape[0]
        fx = P_gl[:,0,0] * (width / 2.0)
        fy = P_gl[:,1,1] * (height / 2.0)
        cx = (1 + P_gl[:,0,2]) * (width / 2.0)
        cy = (1 + P_gl[:,1,2]) * (height / 2.0)

        K = torch.zeros((B, 3, 3), device=P_gl.device, dtype=P_gl.dtype)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0

        return K
        

    # def scale_pmat(self, pmat, scaling_factor):
    #     """
    #     pmat: (B, 3, 4) torch.Tensor, projection matrix
    #     scaling_factor: int, 스케일링 팩터
    #     반환값: (B, 3, 3) @ (B, 3, 4) = (B, 3, 4) torch.Tensor, 스케일링된 프로젝션 매트릭스
    #     """
    #     scaling_matrix = torch.eye(3, device=pmat.device, dtype=pmat.dtype).unsqueeze(0).repeat(pmat.shape[0], 1, 1)
    #     scaling_matrix[:, 0, 0] *= scaling_factor
    #     scaling_matrix[:, 1, 1] *= scaling_factor
    #     scaled_pmat = torch.bmm(scaling_matrix, pmat)  # (B, 3, 4)
    #     return scaled_pmat

    def uppertri_to_symm(self, upper_triangular):
        """
        6D upper triangular covariance를 3x3 대칭 행렬로 변환
        :param upper_triangular: (B, 6) torch.Tensor, 6D upper triangular covariance
        :return: (B, 3, 3) torch.Tensor, 대칭 행렬
        """
        B = upper_triangular.shape[0]
        symm_matrix = torch.zeros((B, 3, 3), device=upper_triangular.device)
        symm_matrix[:, 0, 0] = upper_triangular[:, 0]
        symm_matrix[:, 0, 1] = upper_triangular[:, 1]
        symm_matrix[:, 0, 2] = upper_triangular[:, 2]
        symm_matrix[:, 1, 0] = upper_triangular[:, 1]
        symm_matrix[:, 1, 1] = upper_triangular[:, 3]
        symm_matrix[:, 1, 2] = upper_triangular[:, 4]
        symm_matrix[:, 2, 0] = upper_triangular[:, 2]
        symm_matrix[:, 2, 1] = upper_triangular[:, 4]
        symm_matrix[:, 2, 2] = upper_triangular[:, 5]
        return symm_matrix

    def camera_ids_to_indices(self):
        colmap_point_id_to_index = {point3D.id: idx for idx, (point3D_id, point3D) in enumerate(self.dataset.augmentor.colmap_points3D.items())}
        colmap_image_id_names = {id: colmap_image.name for (id, colmap_image) in self.dataset.augmentor.colmap_images.items()}
        camera_name_index = {camera.image_name: idx for (idx, camera) in enumerate(self.cameras)}
        # print(colmap_point_id_to_index, "colmap_point_id_to_index")
        # print(colmap_image_id_names, "colmap_image_id_names")
        # print(camera_name_index, "camera_name_index")
        # colmap_image_id_to_camera_index 생성 (존재하는 경우만)
        colmap_image_id_to_camera_index = {}
        for id, name in colmap_image_id_names.items():
            if name in camera_name_index.keys():
                colmap_image_id_to_camera_index[id] = camera_name_index[name]
            # else: 없는 경우는 건너뜀

        # p3d_image_id_matrix에서 매칭 안되는 경우 삭제 (역순으로)
        to_delete = []
        for i, (colmap_point_id, colmap_image_id, _, _) in enumerate(self.dataset.p3d_image_id_matrix):
            if colmap_image_id not in colmap_image_id_to_camera_index.keys():
                to_delete.append(i)
            else:
                self.dataset.p3d_image_id_matrix[i][1] = colmap_image_id_to_camera_index[colmap_image_id]
                # self.dataset.p3d_image_id_matrix[i][0] = colmap_point_id_to_index[colmap_point_id]
        print(f"Deleted {len(to_delete)} indices with test cameras")
        # # 역순으로 삭제
        # for i in reversed(to_delete):
        #     del self.dataset.p3d_image_id_matrix[i]
        #     del self.dataset.corresponding_cov_matrix[i]
        #     del self.dataset.corresponding_rgb_values[i]
        self.colmap_image_id_to_camera_index = colmap_image_id_to_camera_index

    def compute_3d_covariance_batch(self, scaling_batch, quaternion_batch):
        R_mm_S = build_scaling_rotation(scaling_batch, quaternion_batch)  # (B, 3, 3)
        cov_batch = R_mm_S @ R_mm_S.transpose(1, 2)
        return cov_batch  # (B, 3, 3)

    def project_cov3d_to_2d_batch(self, cov3d_batch, pmat_batch, xyz_batch):
        """
        cov3d_batch: (B, 3, 3) torch.Tensor, 배치 단위 3D covariance matrix
        pmat_batch: (B, 3, 4) torch.Tensor, 배치 단위 projection matrix
        xyz_batch: (B, 3) torch.Tensor, 배치 단위 3D points
        반환값: (B, 2, 2) torch.Tensor, 배치 단위 2D covariance matrix
        """
        # Jacobian 계산
        J_batch = self.compute_jacobian_batch(pmat_batch, xyz_batch)  # (B, 2, 3)

        # 3D covariance를 2D로 변환
        cov2d_batch = torch.bmm(torch.bmm(J_batch, cov3d_batch), J_batch.transpose(1, 2))  # (B, 2, 2)

        return cov2d_batch

    def kl_divergence_2d_gaussian_batch(self, cov1, cov2):
        """
        cov1: (B, 2, 2) torch.Tensor, 예측 2D covariance
        cov2: (B, 2, 2) torch.Tensor, GT 2D covariance
        반환값: (B,) torch.Tensor, 배치별 KL divergence
        inv 없이 Cholesky 분해로 KL divergence 계산 (수치적으로 더 안정적)
        """
        try:
            # Cholesky 분해 (cov2 = L @ L^T)
            L = torch.linalg.cholesky(cov2)  # (B, 2, 2)
        except torch._C._LinAlgError as e:
            # 에러 발생 시 문제를 일으킨 배치 요소를 확인
            print("Cholesky 분해 실패: 양의 정부호 조건을 만족하지 않는 행렬이 있습니다.")
            pdb.set_trace()
            for i in range(cov2.shape[0]):
                try:
                    torch.linalg.cholesky(cov2[i])  # 개별 행렬 확인
                except torch._C._LinAlgError:
                    print(f"문제 발생한 배치 인덱스: {i}")
                    print(f"cov2[{i}]:\n{cov2[i]}")
                    print(f"cov1[{i}]:\n{cov1[i]}")
                    eigvals = torch.linalg.eigvalsh(cov2[i])
                    print(f"cov2[{i}] 고유값: {eigvals}")
            # 디버깅 정보를 출력한 후 에러를 다시 발생시킴
            raise e  # `e`는 여기서 정의됨

        # Solve L @ X = cov1 for X, then X @ L^T = cov2^{-1} @ cov1
        # 즉, solve(L, cov1) = Y, solve(L^T, Y^T).T = cov2^{-1} @ cov1
        Y = torch.linalg.solve(L, cov1)
        X = torch.linalg.solve(L.transpose(-2, -1), Y.transpose(-2, -1)).transpose(-2, -1)
        trace_term = X.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        det1 = torch.clamp(torch.linalg.det(cov1), min=1e-6)
        det2 = torch.clamp(torch.linalg.det(cov2), min=1e-6)
        log_det_term = torch.log(det2 / det1)
        return 0.5 * (trace_term - 2 + log_det_term)

    def kl_divergence_2d_gaussian(self, cov1, cov2):
        """
        2D 가우시안 분포 간의 KL 발산 계산
        :param cov1: 첫 번째 가우시안의 공분산 행렬 (2x2)
        :param cov2: 두 번째 가우시안의 공분산 행렬 (2x2)
        :return: KL 발산 값
        """
        det1 = torch.det(cov1)
        det2 = torch.det(cov2)
        inv_cov2 = torch.inverse(cov2)
        
        trace_term = torch.trace(torch.mm(inv_cov2, cov1))
        log_det_term = torch.log(det2 / det1)
        
        return 0.5 * (trace_term - 2 + log_det_term)

    def project_cov3d_to_2d(self, cov3d, camera_id):
        # 3D 공분산 행렬을 2D로 투영
        camera = self.cameras[camera_id]
        proj_matrix = camera.full_proj_transform
        cov2d = proj_matrix @ cov3d @ proj_matrix.T
        return cov2d

    def compute_3d_covariance(self, scaling, quaternion):
        """
        scaling: (3,) torch.Tensor, 각 축의 scaling (sx, sy, sz)
        quaternion: (4,) torch.Tensor, (w, x, y, z) 순서
        build_rotation: 쿼터니언을 (3,3) 회전행렬로 변환하는 함수
        반환값: (3,3) torch.Tensor, 3D covariance matrix
        """
        # scaling은 row vector (sx, sy, sz)
        # scaling을 대각행렬로 변환 (row first order)
        S = torch.diag(scaling)  # (3,3)
        # 회전행렬 생성
        R = build_rotation(quaternion)  # (3,3)
        # 공분산 행렬 계산: cov = R @ S @ S @ R.T
        # (row first order, 즉 v' = R @ v)
        cov = R @ S @ S @ R.T
        return cov
    
    def compute_3d_covariance_batch(self, scaling_batch, quaternion_batch):
        """
        scaling_batch: (B, 3) torch.Tensor, 각 축의 scaling (sx, sy, sz)
        quaternion_batch: (B, 4) torch.Tensor, (w, x, y, z) 순서
        반환값: (B, 3, 3) torch.Tensor, 배치 단위 3D covariance matrix
        """
        # scaling을 대각행렬로 변환 (B, 3, 3)
        B = scaling_batch.shape[0]
        S_batch = torch.zeros(B, 3, 3, device=scaling_batch.device, dtype=scaling_batch.dtype)
        S_batch[:, 0, 0] = scaling_batch[:, 0]
        S_batch[:, 1, 1] = scaling_batch[:, 1]
        S_batch[:, 2, 2] = scaling_batch[:, 2]

        # 회전행렬 생성 (B, 3, 3)
        R_batch = build_rotation(quaternion_batch)  # (B, 3, 3)

        # 공분산 행렬 계산: cov = R @ S @ S @ R^T
        SS_batch = torch.bmm(S_batch, S_batch)  # (B, 3, 3)
        temp_batch = torch.bmm(R_batch, SS_batch)  # (B, 3, 3)
        cov_batch = torch.bmm(temp_batch, R_batch.transpose(1, 2))  # (B, 3, 3)

        return cov_batch
    
    def frobenius_norm_batch(self, cov1_batch, cov2_batch):
        """
        cov1_batch: (B, 3, 3) torch.Tensor, 첫 번째 공분산 행렬 배치
        cov2_batch: (B, 3, 3) torch.Tensor, 두 번째 공분산 행렬 배치
        반환값: (B,) torch.Tensor, 각 배치에 대한 Frobenius norm
        """
        diff = cov1_batch - cov2_batch
        return torch.norm(diff, p='fro', dim=(1, 2))

    def compute_jacobian_batch(self, pmat_batch, xyz_batch):
        """
        pmat_batch: (B, 3, 4) torch.Tensor, 배치 단위 projection matrix (4x4)
        xyz_batch: (B, 3) torch.Tensor, 배치 단위 3D points
        반환값: (B, 2, 3) torch.Tensor, 배치 단위 Jacobian matrix
        """
        B = xyz_batch.shape[0]

        ones = torch.ones((B, 1), dtype=xyz_batch.dtype, device=xyz_batch.device)
        Xh = torch.cat([xyz_batch, ones], dim=1)  # (B, 4)
        
        u = torch.sum(pmat_batch[:, 0]*Xh, dim=1)
        v = torch.sum(pmat_batch[:, 1]*Xh, dim=1)
        w = torch.sum(pmat_batch[:, 2]*Xh, dim=1)

        eps=1e-8
        w_safe = w + eps

        x = u / w_safe
        y = v / w_safe

        p1 = pmat_batch[:, 0, :3]
        p2 = pmat_batch[:, 1, :3]
        p3 = pmat_batch[:, 2, :3]

        dxdX = (p1 - (x[:, None] * p3)) / w_safe[:, None]
        dydX = (p2 - (y[:, None] * p3)) / w_safe[:, None]

        J = torch.stack([dxdX, dydX], dim=1)  # (B, 2, 3)
        # # Homogeneous coordinates로 변환
        # points_h_batch = torch.cat([xyz_batch, torch.ones(B, 1, device=xyz_batch.device)], dim=1)  # (B, 4)

        # # 4x4 projection matrix 정규화 (마지막 원소가 1이 아닐 수도 있음)
        # pmat_batch_norm = pmat_batch.clone()
        # last_elem = pmat_batch[:, 3, 3].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        # # 0 division 방지
        # last_elem[last_elem == 0] = 1.0
        # pmat_batch_norm = pmat_batch / last_elem

        # # 4x4 projection matrix에서 상위 3x4만 사용 (일반적인 카메라 프로젝션)
        # pmat3x4_batch = pmat_batch_norm[:, :3, :4]  # (B, 3, 4)

        # # 사영된 2D 좌표 계산
        # projected_batch = torch.bmm(pmat3x4_batch, points_h_batch.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        # x_batch, y_batch, w_batch = projected_batch[:, 0], projected_batch[:, 1], projected_batch[:, 2]

        # # Jacobian 텐서 초기화
        # J_batch = torch.zeros(B, 2, 3, device=pmat_batch.device)  # (B, 2, 3)

        # # 각 요소 계산 및 할당 (연산 그래프 유지)
        # J_batch[:, 0, 0] = pmat3x4_batch[:, 0, 0] / w_batch - x_batch * pmat3x4_batch[:, 2, 0] / (w_batch ** 2)
        # J_batch[:, 0, 1] = pmat3x4_batch[:, 0, 1] / w_batch - x_batch * pmat3x4_batch[:, 2, 1] / (w_batch ** 2)
        # J_batch[:, 0, 2] = pmat3x4_batch[:, 0, 2] / w_batch - x_batch * pmat3x4_batch[:, 2, 2] / (w_batch ** 2)
        # J_batch[:, 1, 0] = pmat3x4_batch[:, 1, 0] / w_batch - y_batch * pmat3x4_batch[:, 2, 0] / (w_batch ** 2)
        # J_batch[:, 1, 1] = pmat3x4_batch[:, 1, 1] / w_batch - y_batch * pmat3x4_batch[:, 2, 1] / (w_batch ** 2)
        # J_batch[:, 1, 2] = pmat3x4_batch[:, 1, 2] / w_batch - y_batch * pmat3x4_batch[:, 2, 2] / (w_batch ** 2)

        # return J_batch
        return J

# class QuadInitGaussianModel(GaussianModel):
#     def __init__(self, xyz, features, scales, rots, opacities, camera_dict):
#         super().__init__()
#         self.camera_dict = camera_dict

#     def project_gaussian_to_image(self, point_idx, camera_id):
#         pass
    
#     def forward(self, point_idx, camera_id):
#         # 예시: 파라미터를 그대로 반환
#         mu_2d, cov_2d = self.project_gaussian_to_image(point_idx, camera_id)
#         rgb = self.get_rgb(point_idx)
#         return cov_2d, rgb

# # 예시 최적화 루프
# def optimize_model(model, loss_fn, n_steps=100, lr=1e-3):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     for step in range(n_steps):
#         optimizer.zero_grad()
#         params = model()
#         # 사용자 정의 loss 계산
#         loss = loss_fn(params)
#         loss.backward()
#         optimizer.step()
#         if step % 10 == 0:
#             print(f"Step {step}: loss = {loss.item():.6f}")

class QuadtreeInitDataset(Dataset):
    def __init__(self, image_path, colmap_path):
        self.image_path = image_path
        self.colmap_path = colmap_path
        augmentor_dir = os.path.abspath(os.path.join(image_path, os.pardir))
        augmentor_path = os.path.join(augmentor_dir, "augmentor.pkl")
        import pickle
        if os.path.exists(augmentor_path):
            print(f"Loading existing augmentor from {augmentor_path}")
            self.augmentor = pickle.load(open(augmentor_path, "rb"))
        else:
            print(f"Creating new augmentor for {image_path} and {colmap_path}")
            self.augmentor = Augmentor(image_path, colmap_path, mode="full")
            with open(augmentor_path, "wb") as f:
                pickle.dump(self.augmentor, f)
            print(f"Augmentor saved to {augmentor_path}")
        # self.augmentor = Augmentor(image_path, colmap_path, mode="decompose")
        self.image_scale = int(image_path.split("/")[-1].split("_")[-1])
        self.scale = 5
        print(f"Scale set to {self.scale}! This is used to scale the covariance matrix.")
        self.filter_images_sobel_gaussian()
        self.create_p3d_image_id_matrix()
        # pdb.set_trace()
        self.mode = "sfm"
        self.n_colmap_points = len(self.augmentor.colmap_points3D)

    def sobel_gaussian(self, image):
        gauss = cv2.GaussianBlur(image, (5, 5), 0)
        sobel_x = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=5)
        return sobel_x, sobel_y

    def filter_images_sobel_gaussian(self):
        images = {image_id: cv2.cvtColor(cv2.imread(os.path.join(self.augmentor.image_dir, self.augmentor.colmap_images[image_id].name)), cv2.COLOR_BGR2GRAY)/255. for image_id in self.augmentor.image_keys}
        ixs = {}
        iys = {}
        for image_id, image in images.items():
            sobel_x, sobel_y = self.sobel_gaussian(image)
            ixs[image_id] = sobel_x
            iys[image_id] = sobel_y
        self.ixs = ixs
        self.iys = iys

    def gradient_to_covariance(self, image_id, point2D_coord, corresponding_width):
        point2D_coordx, point2D_coordy = point2D_coord
        ix = self.ixs[image_id][point2D_coordy, point2D_coordx]
        iy = self.iys[image_id][point2D_coordy, point2D_coordx]
        if ix < 1e-3 and iy < 1e-3:
            cov_matrix = np.asarray([[corresponding_width**2/36, 0], [0, corresponding_width**2/36]])
            return cov_matrix, None
        elif ix < 1e-3 and iy >= 1e-3:
            cov_matrix = np.asarray([[corresponding_width**2/36/9, 0], [0, corresponding_width**2/36]])
            return cov_matrix, None
        elif ix >= 1e-3 and iy < 1e-3:
            cov_matrix = np.asarray([[corresponding_width**2/36, 0], [0, corresponding_width**2/36/9]])
            return cov_matrix, None
        else:
            structure_tensor = np.array([[ix**2, ix*iy], [ix*iy, iy**2]]) + 1e-6  # Add small value to avoid singularity
            eigvals, eigvecs = np.linalg.eigh(structure_tensor)
            eigvs = eigvals
            eigvals_max = np.max(eigvals)
            eigvals = np.maximum(eigvals/eigvals_max, 0.2)
            eigvals = np.clip(eigvals, 0.2, 1.0)  # Ensure eigenvalues are not too small
            eigvals = np.sqrt(eigvals) * corresponding_width / 6
            scale_matrix = np.diag([eigvals[1], eigvals[0]])
            cov_matrix = eigvecs @ scale_matrix @ scale_matrix @ eigvecs.T
        return cov_matrix, eigvs

    def create_p3d_image_id_matrix(self):
        # p3d_image_id_matrix = list([point3D.id, image_id, point2D_coordx, point2D_coordy])
        self.p3d_image_id_matrix = []
        self.corresponding_cov_matrix = []
        # self.corresponding_rgb_values = []
        self.test_only_3d_indices = []
        from tqdm import tqdm
        for point3D_idx, (point3D_id, point3D) in tqdm(enumerate(self.augmentor.colmap_points3D.items())):
            for i, image_id in enumerate(point3D.image_ids):
                if image_id not in self.augmentor.image_keys:
                    self.test_only_3d_indices.append(point3D_idx)
                    continue
                point2D_idx = point3D.point2D_idxs[i]
                point2D_coordx, point2D_coordy = self.augmentor.colmap_images[image_id].xys[point2D_idx]
                self.p3d_image_id_matrix.append([point3D_idx, image_id, None, None])
                point2D_coordx = np.clip(int(point2D_coordx/self.image_scale), 0, self.augmentor.roots[image_id].width-1)
                point2D_coordy = np.clip(int(point2D_coordy/self.image_scale), 0, self.augmentor.roots[image_id].height-1)
                # self.p3d_image_id_matrix[-1][2] = point2D_coordx
                # self.p3d_image_id_matrix[-1][3] = point2D_coordy
                corresponding_leaf_node = self.augmentor.find_corresponding_leaf_node(image_id,
                                                                                    [point2D_coordx,
                                                                                    point2D_coordy])
                corresponding_width = corresponding_leaf_node.width
                corresponding_height = corresponding_leaf_node.height
                if corresponding_leaf_node is None:
                    print(f"Warning: No corresponding leaf node found for image {image_id} at ({point2D_coordx}, {point2D_coordy})")
                    self.p3d_image_id_matrix.pop()
                    continue
                cov_matrix, eigvs = self.gradient_to_covariance(image_id, [point2D_coordx, point2D_coordy], corresponding_width)
                # cov_matrix = [[corresponding_width**2/36*self.scale, 0], [0, corresponding_height**2/36*self.scale]]
                if cov_matrix is None:
                    self.p3d_image_id_matrix.pop()
                    continue
                if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
                    print(f"Warning: NaN or Inf in covariance matrix for image {image_id} at ({point2D_coordx}, {point2D_coordy})")
                    print(f"eigenvalues: {eigvs}")
                    self.p3d_image_id_matrix.pop()
                    continue
                self.corresponding_cov_matrix.append(cov_matrix * self.scale)

                # rgb_value = corresponding_leaf_node.sampled_point_rgb
                # self.corresponding_rgb_values.append(rgb_value)
                
    
    def augmentation_mode(self):
        # Set dataset to augmentation mode
        self.mode = "augmentation"
        self.p3d_ids = [id + self.n_colmap_points for id in self.augmentor.merged_sample_points_indices]
        self.p3d_image_ids = self.augmentor.merged_image_ids
        # self.p3d_means = torch.tensor(np.asarray(self.augmentor.mean_labels, dtype=np.float32), device="cuda")
        self.p3d_covs = torch.tensor(np.asarray(self.augmentor.cov_labels, dtype=np.float32)*self.scale, device="cuda")
        # pdb.set_trace()
        # self.p3d_rgbs = torch.tensor(np.asarray(self.augmentor.rgb_labels, dtype=np.float32), device="cuda")

        # print(np.array(self.p3d_ids).shape, "p3d_ids shape")
        # print(np.array(self.p3d_image_ids).shape, "p3d_image_ids shape")
        # print(np.array(self.p3d_means).shape, "p3d_means shape")
        # print(np.array(self.p3d_covs).shape, "p3d_covs shape")
        # print(np.array(self.p3d_rgbs).shape, "p3d_rgbs shape")

    def __len__(self):
        if self.mode == "augmentation":
            return len(self.p3d_ids)
        else:
            return len(self.p3d_image_id_matrix)

    def __getitem__(self, idx):
        if self.mode == "augmentation":
            # mean = self.p3d_means[idx]
            cov = self.p3d_covs[idx]
            # rgb = self.p3d_rgbs[idx]
            return (self.p3d_ids[idx], self.p3d_image_ids[idx]),\
                    cov
                #    (mean, cov, rgb)
                    
        else:
            return (self.p3d_image_id_matrix[idx][0], self.p3d_image_id_matrix[idx][1]),\
                   self.corresponding_cov_matrix[idx]
                #    (self.corresponding_cov_matrix[idx], self.corresponding_rgb_values[idx])

# 사용 예시
if __name__ == "__main__":
    # 임의 데이터 예시
    N, C = 10, 16
    xyz = np.random.randn(N, 3)
    features = np.random.randn(N, C)
    scales = np.abs(np.random.randn(N, 3))
    rots = np.random.randn(N, 4)
    opacities = np.random.rand(N, 1)

    model = GaussianModel(xyz, features, scales, rots, opacities)

    # 예시 loss 함수 (사용자 정의로 대체)
    def dummy_loss(params):
        return params["xyz"].norm() + params["scaling"].norm()

    optimize_model(model, dummy_loss, n_steps=50)