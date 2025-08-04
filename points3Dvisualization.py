import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

class Colmap3DPoint:
    def __init__(self, imageids, imagenames, imagepath, point2Dcoords, cov2Dmatrices, point3Dcoord, cov3Dmatrix, intrinsic_matrices, rotation_matrices, translation_vectors):
        self.imageids = imageids
        self.imagenames = imagenames
        self.imagepath = imagepath
        self.point2Dcoords = point2Dcoords
        self.cov2Dmatrices = cov2Dmatrices
        self.point3Dcoord = point3Dcoord
        self.cov3Dmatrix = cov3Dmatrix
        self.intrinsic_matrices = intrinsic_matrices
        self.rotation_matrices = rotation_matrices
        self.translation_vectors = translation_vectors
        self.num_images = len(imageids)

    def visualize_methods(self, method_num=0):
        uvs, proj_cov2Ds = self.project3D2D()
        images = [cv2.cvtColor(cv2.imread(os.path.join(self.imagepath, imagename)), cv2.COLOR_BGR2RGB) for imagename in self.imagenames]
        # method 1. Clip lambda with min bound 1e-3
        def compute_cov2D_method1(eigenvalues, min_bound=1e-3):
            eigenvalues = np.clip(eigenvalues, min_bound, None)
            s = eigenvalues[0] + min_bound
            return np.diag(s/(eigenvalues+min_bound))

        # method 2. Threshold with divergence of gradient
        def compute_cov2D_method2(ix, iy, eigenvalues, threshold=0.1):
            grad_image = np.asarray([[ix, iy]])
            grad_magnitude = np.linalg.norm(grad_image)
            if grad_magnitude < threshold:
                return np.diag([1., 1.])
            else:
                s = eigenvalues[0] + 1e-3
                return np.diag(s/(eigenvalues + 1e-3))

        # method 3. Softmax with sigmoid of divergence of gradient
        def compute_cov2D_method3(ix, iy, eigenvalues):
            grad_image = np.asarray([[ix, iy]])
            grad_magnitude = np.linalg.norm(grad_image)
            alpha = 1 / (1 + np.exp(-grad_magnitude))
            s = eigenvalues[0] + 1e-3
            scale_matrix = alpha * np.diag(s/(eigenvalues + 1e-3)) + (1 - alpha) * np.diag([1., 1.])
            return scale_matrix / np.max(scale_matrix)
        if method_num != 0:
            uv_ref = self.point2Dcoords
            ixs = [cv2.Sobel(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)/255.0, (5,5), 0), cv2.CV_64F, 1, 0, ksize=5)[uv[1],uv[0]] for image, uv in zip(images, uv_ref)]
            iys = [cv2.Sobel(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)/255.0, (5,5), 0), cv2.CV_64F, 0, 1, ksize=5)[uv[1],uv[0]] for image, uv in zip(images, uv_ref)]
            structure_tensors = [np.asarray([[ix**2, ix * iy], [ix * iy, iy**2]]) for ix, iy in zip(ixs, iys)]
            print([st.shape for st in structure_tensors])
            eigdecs = [np.linalg.eigh(st) for st in structure_tensors]
            eigvals = [eigdec[0] for eigdec in eigdecs]
            eigvecs = [eigdec[1] for eigdec in eigdecs]
            if method_num == 1:
                cov2Ds = [eigvecs[i] @ compute_cov2D_method1(eigvals[i]) @ eigvecs[i].T for i in range(len(eigvals))]
            elif method_num == 2:
                cov2Ds = [eigvecs[i] @ compute_cov2D_method2(ixs[i], iys[i], eigvals[i]) @ eigvecs[i].T for i in range(len(eigvals))]
            elif method_num == 3:
                cov2Ds = [eigvecs[i] @ compute_cov2D_method3(ixs[i], iys[i], eigvals[i]) @ eigvecs[i].T for i in range(len(eigvals))]

        max_cols = 5
        num_images = self.num_images
        n_cols = min(num_images, max_cols)
        n_rows = math.ceil(num_images / max_cols)

        # 이미지 크기에 맞춰 figure 크기 조정
        img_height, img_width = images[0].shape[:2]
        aspect_ratio = img_width / img_height
        
        # 더 작은 subplot 크기와 조정된 figure 크기
        subplot_width = 4
        subplot_height = subplot_width / aspect_ratio
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols * subplot_width, n_rows * subplot_height),
                            squeeze=False)
        
        # 서브플롯 간 간격 줄이기
        fig.subplots_adjust(
            left=0.02,    # 왼쪽 여백
            bottom=0.02,  # 아래쪽 여백
            right=0.98,   # 오른쪽 여백
            top=0.95,     # 위쪽 여백
            wspace=0.05,  # 열 간격
            hspace=0.15   # 행 간격 (제목 공간 고려)
        )
        
        for i in range(num_images):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            uv_main = uvs[i]
            cov_main = proj_cov2Ds[i]
            uv_ref = self.point2Dcoords[i]
            if method_num != 0:
                cov_ref = cov2Ds[i]
            else:
                cov_ref = self.cov2Dmatrices[i]

            ax.imshow(images[i])
            ax.axis('off')
            ax.set_title(f"Image {i}", fontsize=10, pad=5)

            # 원본 위에 ellipse 그리기
            self.drawEllipseCov2D(ax, uv_main, cov_main, edgecolor='lime', linestyle='-')
            self.drawEllipseCov2D(ax, uv_ref, cov_ref, edgecolor='blue', linestyle='--')

            # ====== 확대 영역 ======
            zoom_size = 5
            x0, y0 = uv_main
            x1, x2 = np.round(x0 - zoom_size), np.round(x0 + zoom_size)
            y1, y2 = np.round(y0 - zoom_size), np.round(y0 + zoom_size)

            # inset axes 크기 조정
            axins = inset_axes(ax, width="25%", height="25%", loc='upper right', borderpad=0.5)

            axins.imshow(images[i])
            axins.set_xlim(x1, x2)
            axins.set_ylim(y2, y1)

            self.drawEllipseCov2D(axins, uv_main, cov_main, edgecolor='lime', linestyle='-')
            self.drawEllipseCov2D(axins, uv_ref, cov_ref, edgecolor='blue', linestyle='--')

            axins.axis('off')
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="cyan", lw=0.5)
        
        # 빈 서브플롯 숨기기
        for i in range(num_images, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.show()

    def compute_depth(self, point3D, imageid):
        rotation_matrix = self.rotation_matrices[imageid]
        translation_vector = self.translation_vectors[imageid].reshape(3, 1)
        point3D_world = point3D.reshape(3, 1)
        point3D_camera = np.einsum('ij,jk->ik', rotation_matrix, point3D_world) + translation_vector
        depth = point3D_camera[2, 0]
        return depth
    
    def compute_scale_size(self, depths):
        depths = np.asarray(depths)
        mean_depth = np.mean(depths)
        return mean_depth / depths
    
    def computeJacobianMatrices(self, point3Dcoord, imageids, intrinsic_matrices, rotation_matrices, translation_vectors):
        """
        Computes the Jacobian matrices for the 3D points based on the provided parameters.
        """
        n_images = len(imageids)
        Rs = np.asarray([
            rotation_matrices[imageid] for imageid in imageids
        ]).reshape(n_images, 3, 3)
        ts = np.asarray([
            translation_vectors[imageid] for imageid in imageids
        ]).reshape(n_images, 3, 1)
        Ks = np.asarray([
            intrinsic_matrices[1] for _ in imageids
        ]).reshape(n_images, 3, 3)
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

    def vech(self, matrix):
        n = matrix.shape[0]
        if matrix.shape[1] == matrix.shape[2] == 2:
            vech_matrix = np.zeros((n, 3))
            vech_matrix[:, 0] = matrix[:, 0, 0]
            vech_matrix[:, 1] = matrix[:, 1, 1]
            vech_matrix[:, 2] = matrix[:, 0, 1]
        return vech_matrix

    def unvech(self, vech_matrix):
        assert vech_matrix.reshape(-1).shape[0] == 6
        matrix = np.zeros((3, 3))
        matrix[0, 0] = vech_matrix[0]
        matrix[1, 1] = vech_matrix[1]
        matrix[2, 2] = vech_matrix[2]
        matrix[0, 1] = vech_matrix[3]
        matrix[1, 0] = vech_matrix[3]
        matrix[0, 2] = vech_matrix[4]
        matrix[2, 0] = vech_matrix[4]
        matrix[1, 2] = vech_matrix[5]
        matrix[2, 1] = vech_matrix[5]
        return matrix

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

    def solveLeastSquaresCov2D3D(self):
        jacobian_matrices = self.computeJacobianMatrices(self.point3Dcoord,
                                                    self.imageids,
                                                    self.intrinsic_matrices,
                                                    self.rotation_matrices,
                                                    self.translation_vectors)
        
        cov2Dmatrices = np.asarray([
            self.cov2Dmatrices
        ]).reshape(len(self.imageids), 2, 2)
        
        vech_c2D = self.vech(cov2Dmatrices)
        
        J = self.kroneckerJacobianMatrices(jacobian_matrices)

        b = vech_c2D.reshape(-1, 1)
        A = J
        assert b.shape == (len(self.imageids)*3, 1)
        assert A.shape == (len(self.imageids)*3, 6)
        x_est = np.linalg.lstsq(A, b, rcond=None)[0]
        Sigma3D_est = np.array([
            self.unvech(x_est)
        ])
        return Sigma3D_est
    
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
    
    def sigma3D_to_vech(self, Sigma):
        return np.array([
            Sigma[0, 0], Sigma[1, 1], Sigma[2, 2], Sigma[0, 1], Sigma[0, 2], Sigma[1, 2]
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
    
    def solveCholeskyMinimizeCov2D3D(self, cov2Dmat = None):
        from scipy.optimize import minimize
        jacobian_matrices = self.computeJacobianMatrices(self.point3Dcoord,
                                                    self.imageids,
                                                    self.intrinsic_matrices,
                                                    self.rotation_matrices,
                                                    self.translation_vectors)
        if cov2Dmat is not None:
            cov2Dmatrices = np.asarray([
                cov2Dmat]).reshape(len(self.imageids), 2, 2)
        else:
            cov2Dmatrices = np.asarray([
                self.cov2Dmatrices
            ]).reshape(len(self.imageids), 2, 2)
            
        vech_c2D = self.vech(cov2Dmatrices)
        
        J = self.kroneckerJacobianMatrices(jacobian_matrices)

        init_params = np.array([0., 0., 0., 0., 0., 0.])
        result = minimize(self.loss_fn_cholesky, init_params, args=(J, vech_c2D), method='L-BFGS-B')

        Sigma3D_est = self.cholesky_to_sigma3D(result.x)
        return Sigma3D_est

    def visualizeAdaptivePatching(self, patch_length=0.5):
        proj = self.project3D2D()
        uvs = proj["point3D_uv"]
        proj_cov2Ds = proj["cov3Dmatrix_proj"]
        depths = proj["point3D_depth"]
        images = [cv2.cvtColor(cv2.imread(os.path.join(self.imagepath, imagename)), cv2.COLOR_BGR2RGB) for imagename in self.imagenames]
        max_cols = 5
        num_images = self.num_images
        n_cols = min(num_images, max_cols)
        n_rows = math.ceil(num_images / max_cols)
        ################################################
        # Adaptive Patching
        # GaussianFilter -> SobelFilter -> MeanFilter -> StructureTensor -> EigenDecomposition -> Label Generation
        Ixs = [cv2.Sobel(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)/255.0, (5,5), 0), cv2.CV_64F, 1, 0, ksize=5) for image in images]
        Iys = [cv2.Sobel(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)/255.0, (5,5), 0), cv2.CV_64F, 0, 1, ksize=5) for image in images]
        # Determine the window size for the mean filter with depth
        depth_mean = np.mean(depths)
        focal_length_mean = np.mean([self.intrinsic_matrices[1][0, 0], self.intrinsic_matrices[1][1, 1]])
        window_sizes = np.floor(focal_length_mean * patch_length / depths.reshape(-1))
        # window_sizes = np.floor(depth_mean / depths.reshape(-1) * 5)
        
        # Ensure window sizes are at least 3 and odd
        window_sizes = np.maximum(window_sizes, 3)
        window_sizes = (window_sizes // 2) * 2 + 1
        window_sizes = window_sizes.astype(int)
        print(window_sizes)
        Ix_means = [cv2.GaussianBlur(Ix, (int(window_size), int(window_size)), 0)[uvs[i][1], uvs[i][0]] for i, (Ix, window_size) in enumerate(zip(Ixs, window_sizes))]
        Iy_means = [cv2.GaussianBlur(Iy, (int(window_size), int(window_size)), 0)[uvs[i][1], uvs[i][0]] for i, (Iy, window_size) in enumerate(zip(Iys, window_sizes))]
        structure_tensors = [np.asarray([[Ix**2, Ix * Iy], [Ix * Iy, Iy**2]]) for Ix, Iy in zip(Ix_means, Iy_means)]
        print(structure_tensors[0].shape)
        eigdecs = [np.linalg.eigh(st.reshape(2,2)) for st in structure_tensors]
        eigvals = [eigdec[0] for eigdec in eigdecs]
        eigvecs = [eigdec[1] for eigdec in eigdecs]
        adapatch_cov2Ds = []
        for i in range(len(eigvals)):
            grad_magnitude = np.linalg.norm(np.asarray([Ix_means[i], Iy_means[i]]))
            alpha = 1 / (1 + np.exp(-grad_magnitude))

            rel_eigvals = eigvals[i] + 1e-6
            rel_scale = (np.min(rel_eigvals) / rel_eigvals)
            # Ensure the relative scale is not too small
            rel_scale = np.clip(rel_scale, 0.1, 1)
            max_scale = (window_sizes[i] / 2) ** 2
            anisotropic_part = np.diag(max_scale * rel_scale)
            isotropic = np.diag([max_scale, max_scale])
            scale_matrix = alpha * anisotropic_part + (1 - alpha) * isotropic
            eigvals_tmp, eigvecs_tmp = np.linalg.eigh(scale_matrix)
            k = max_scale / np.max(eigvals_tmp)
            scale_matrix = scale_matrix * k

            cov2D = eigvecs[i] @ scale_matrix @ eigvecs[i].T

            adapatch_cov2Ds.append(cov2D)
        ################################################

        Sigma3D_est = self.solveCholeskyMinimizeCov2D3D(adapatch_cov2Ds)

        eigval, eigvec = np.linalg.eigh(Sigma3D_est.reshape(3, 3))
        # Rescale covariances w.r.t. patch length
        # Recall that eigenvalues are sorted in ascending order
        # scale_factor = patch_length / (2 * np.sqrt(eigval[2]))
        # Rather than scaling with patch length in 3D space directly,
        # determine scale factor based on the projected 2D covariance
        proj_temp = self.project3D2D(Sigma3D_est.reshape(3, 3))
        sfs = []
        for i, cov2d_temp in enumerate(proj_temp["cov3Dmatrix_proj"]):
            eigval_temp, eigvec_temp = np.linalg.eigh(cov2d_temp)
            # We should rescale the principal axis to match the window size
            target_size = window_sizes[i] / 2
            sf = target_size / np.sqrt(eigval_temp[1])
            sfs.append(sf)

        scale_factor = np.mean(np.asarray(sfs))
        Sigma3D_est = eigvec @ np.diag(eigval * scale_factor**2) @ eigvec.T
        print("Sigma3D_est", Sigma3D_est, sep='\n')
        print("Eigenvalues", eigval, sep='\n')
        print("scale_factor", scale_factor)
        print("Eigenvalues after rescaling", eigval * scale_factor**2, sep='\n')
        print("Sigma3D_est after rescaling", Sigma3D_est, sep='\n')
        

        newproj = self.project3D2D(Sigma3D_est.reshape(3, 3))
        uvs = newproj["point3D_uv"]
        proj_cov2Ds_est = newproj["cov3Dmatrix_proj"]
        depths = newproj["point3D_depth"]

        # 이미지 크기에 맞춰 figure 크기 조정
        img_height, img_width = images[0].shape[:2]
        aspect_ratio = img_width / img_height
        
        # 더 작은 subplot 크기와 조정된 figure 크기
        subplot_width = 4
        subplot_height = subplot_width / aspect_ratio
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols * subplot_width, n_rows * subplot_height),
                            squeeze=False)
        
        # 서브플롯 간 간격 줄이기
        fig.subplots_adjust(
            left=0.02,    # 왼쪽 여백
            bottom=0.02,  # 아래쪽 여백
            right=0.98,   # 오른쪽 여백
            top=0.95,     # 위쪽 여백
            wspace=0.05,  # 열 간격
            hspace=0.15   # 행 간격 (제목 공간 고려)
        )
        
        for i in range(num_images):
            # depth = self.compute_depth(self.point3Dcoord, self.imageids[i])
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            uv_main = uvs[i]
            cov_main = proj_cov2Ds[i]
            uv_ref = self.point2Dcoords[i]
            cov_ref = self.cov2Dmatrices[i]
            
            ax.imshow(images[i])
            ax.axis('off')
            ax.set_title(f"Image {i}, depth={np.round(depths[i], 2)}, window_size={window_sizes[i]}", fontsize=10, pad=5)

            # 원본 위에 ellipse 그리기
            # self.drawEllipseCov2D(ax, uv_main, cov_main, edgecolor='lime', linestyle='-')
            # self.drawEllipseCov2D(ax, uv_ref, cov_ref, edgecolor='blue', linestyle='--')
            self.drawEllipseCov2D(ax, uv_ref, adapatch_cov2Ds[i], edgecolor='magenta', linestyle='-')
            self.drawEllipseCov2D(ax, uv_ref, proj_cov2Ds_est[i], 3, edgecolor='lime', linestyle='--')

            # ====== 확대 영역 ======
            zoom_size = window_sizes[i] // 2 + 1
            x0, y0 = uv_main
            x1, x2 = int(x0 - zoom_size), int(x0 + zoom_size)
            y1, y2 = int(y0 - zoom_size), int(y0 + zoom_size)

            # inset axes 크기 조정
            axins = inset_axes(ax, width="25%", height="25%", loc='upper right', borderpad=0.5)

            axins.imshow(images[i])
            axins.set_xlim(x1, x2)
            axins.set_ylim(y2, y1)

            # self.drawEllipseCov2D(axins, uv_main, cov_main, edgecolor='lime', linestyle='-')
            # self.drawEllipseCov2D(axins, uv_ref, cov_ref, edgecolor='blue', linestyle='--')
            self.drawEllipseCov2D(axins, uv_ref, adapatch_cov2Ds[i], edgecolor='magenta', linestyle='-')
            self.drawEllipseCov2D(axins, uv_ref, proj_cov2Ds_est[i], edgecolor='lime', linestyle='--')

            # Draw rectangle on window size
            rect = plt.Rectangle((x1, y1), 2*zoom_size, 2*zoom_size, linewidth=1, edgecolor='cyan', facecolor='none', linestyle='--')
            axins.add_patch(rect)

            axins.axis('off')
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="cyan", lw=0.5)
        
        # 빈 서브플롯 숨기기
        for i in range(num_images, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.show()

        return {"point3D": self.point3Dcoord,
                "intrinsic_matrices": self.intrinsic_matrices,
                "rotation_matrices": [self.rotation_matrices[image_id] for image_id in self.imageids],
                "translation_vectors": [self.translation_vectors[image_id] for image_id in self.imageids],
                "cov2ds": adapatch_cov2Ds,
                "cov3dmatrix": Sigma3D_est.reshape(3, 3),}

    def visualize_covls(self):
        uvs, proj_cov2Ds, s3d = self.project3D2D()
        Sigma3D_est = self.solveLeastSquaresCov2D3D()
        print(Sigma3D_est.shape)
        print("Sigma3D_ls", Sigma3D_est)
        uvs, proj_estcov2Ds, s3d = self.project3D2D(Sigma3D_est.reshape(3, 3))
        assert s3d is not None
        Sigma3D_est2 = self.solveCholeskyMinimizeCov2D3D()
        print(Sigma3D_est2.shape)
        print("Sigma3D_cholesky", Sigma3D_est2)
        uvs, proj_estcov2Ds2, s3d2 = self.project3D2D(Sigma3D_est2.reshape(3, 3))
        assert s3d2 is not None
        print(proj_estcov2Ds2.shape)
        images = [cv2.cvtColor(cv2.imread(os.path.join(self.imagepath, imagename)), cv2.COLOR_BGR2RGB) for imagename in self.imagenames]
        max_cols = 5
        num_images = self.num_images
        n_cols = min(num_images, max_cols)
        n_rows = math.ceil(num_images / max_cols)

        # 이미지 크기에 맞춰 figure 크기 조정
        img_height, img_width = images[0].shape[:2]
        aspect_ratio = img_width / img_height
        
        # 더 작은 subplot 크기와 조정된 figure 크기
        subplot_width = 4
        subplot_height = subplot_width / aspect_ratio
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols * subplot_width, n_rows * subplot_height),
                            squeeze=False)
        
        # 서브플롯 간 간격 줄이기
        fig.subplots_adjust(
            left=0.02,    # 왼쪽 여백
            bottom=0.02,  # 아래쪽 여백
            right=0.98,   # 오른쪽 여백
            top=0.95,     # 위쪽 여백
            wspace=0.05,  # 열 간격
            hspace=0.15   # 행 간격 (제목 공간 고려)
        )
        depth = []
        for i in range(num_images):
            depth.append(self.compute_depth(self.point3Dcoord, self.imageids[i]))
        
        scales = self.compute_scale_size(depth)
        
        for i in range(num_images):
            # depth = self.compute_depth(self.point3Dcoord, self.imageids[i])
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            uv_main = uvs[i]
            cov_main = proj_cov2Ds[i]
            uv_ref = self.point2Dcoords[i]
            cov_ref = self.cov2Dmatrices[i]
            cov_est = proj_estcov2Ds[i]
            cov_est2 = proj_estcov2Ds2[i]
            
            ax.imshow(images[i])
            ax.axis('off')
            ax.set_title(f"Image {i}, depth={np.round(depth[i], 2)}, scale={np.round(scales[i], 2)}", fontsize=10, pad=5)

            # 원본 위에 ellipse 그리기
            self.drawEllipseCov2D(ax, uv_main, cov_main, edgecolor='lime', linestyle='-')
            self.drawEllipseCov2D(ax, uv_ref, cov_ref, edgecolor='blue', linestyle='--')
            self.drawEllipseCov2D(ax, uv_ref, cov_est, edgecolor='red', linestyle='--')
            self.drawEllipseCov2D(ax, uv_ref, cov_est2, edgecolor='orange', linestyle='-.')
            # self.drawEllipseCov2D(ax, uv_ref, cov_ref * (scales[i]**2), edgecolor='red', linestyle=':')

            # ====== 확대 영역 ======
            zoom_size = 5
            x0, y0 = uv_main
            x1, x2 = int(x0 - zoom_size), int(x0 + zoom_size)
            y1, y2 = int(y0 - zoom_size), int(y0 + zoom_size)

            # inset axes 크기 조정
            axins = inset_axes(ax, width="50%", height="50%", loc='upper right', borderpad=0.5)

            axins.imshow(images[i])
            axins.set_xlim(x1, x2)
            axins.set_ylim(y2, y1)

            self.drawEllipseCov2D(axins, uv_main, cov_main, edgecolor='lime', linestyle='-')
            self.drawEllipseCov2D(axins, uv_ref, cov_ref, edgecolor='blue', linestyle='--')
            self.drawEllipseCov2D(axins, uv_ref, cov_est, edgecolor='red', linestyle='--')
            self.drawEllipseCov2D(axins, uv_ref, cov_est2, edgecolor='orange', linestyle='-.')
            # self.drawEllipseCov2D(axins, uv_ref, cov_ref * (scales[i]**2), edgecolor='red', linestyle=':')

            axins.axis('off')
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="cyan", lw=1.0)
        
        # 빈 서브플롯 숨기기
        for i in range(num_images, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.show()

    def visualize(self):
        uvs, proj_cov2Ds = self.project3D2D()
        images = [cv2.cvtColor(cv2.imread(os.path.join(self.imagepath, imagename)), cv2.COLOR_BGR2RGB) for imagename in self.imagenames]
        max_cols = 5
        num_images = self.num_images
        n_cols = min(num_images, max_cols)
        n_rows = math.ceil(num_images / max_cols)

        # 이미지 크기에 맞춰 figure 크기 조정
        img_height, img_width = images[0].shape[:2]
        aspect_ratio = img_width / img_height
        
        # 더 작은 subplot 크기와 조정된 figure 크기
        subplot_width = 4
        subplot_height = subplot_width / aspect_ratio
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols * subplot_width, n_rows * subplot_height),
                            squeeze=False)
        
        # 서브플롯 간 간격 줄이기
        fig.subplots_adjust(
            left=0.02,    # 왼쪽 여백
            bottom=0.02,  # 아래쪽 여백
            right=0.98,   # 오른쪽 여백
            top=0.95,     # 위쪽 여백
            wspace=0.05,  # 열 간격
            hspace=0.15   # 행 간격 (제목 공간 고려)
        )
        depth = []
        for i in range(num_images):
            depth.append(self.compute_depth(self.point3Dcoord, self.imageids[i]))
        
        scales = self.compute_scale_size(depth)
        
        for i in range(num_images):
            # depth = self.compute_depth(self.point3Dcoord, self.imageids[i])
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            uv_main = uvs[i]
            cov_main = proj_cov2Ds[i]
            uv_ref = self.point2Dcoords[i]
            cov_ref = self.cov2Dmatrices[i]
            
            ax.imshow(images[i])
            ax.axis('off')
            ax.set_title(f"Image {i}, depth={np.round(depth[i], 2)}, scale={np.round(scales[i], 2)}", fontsize=10, pad=5)

            # 원본 위에 ellipse 그리기
            self.drawEllipseCov2D(ax, uv_main, cov_main, edgecolor='lime', linestyle='-')
            self.drawEllipseCov2D(ax, uv_ref, cov_ref, edgecolor='blue', linestyle='--')
            self.drawEllipseCov2D(ax, uv_ref, cov_ref * (scales[i]**2), edgecolor='red', linestyle=':')

            # ====== 확대 영역 ======
            zoom_size = 5
            x0, y0 = uv_main
            x1, x2 = int(x0 - zoom_size), int(x0 + zoom_size)
            y1, y2 = int(y0 - zoom_size), int(y0 + zoom_size)

            # inset axes 크기 조정
            axins = inset_axes(ax, width="25%", height="25%", loc='upper right', borderpad=0.5)

            axins.imshow(images[i])
            axins.set_xlim(x1, x2)
            axins.set_ylim(y2, y1)

            self.drawEllipseCov2D(axins, uv_main, cov_main, edgecolor='lime', linestyle='-')
            self.drawEllipseCov2D(axins, uv_ref, cov_ref, edgecolor='blue', linestyle='--')
            self.drawEllipseCov2D(axins, uv_ref, cov_ref * (scales[i]**2), edgecolor='red', linestyle=':')

            axins.axis('off')
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="cyan", lw=0.5)
        
        # 빈 서브플롯 숨기기
        for i in range(num_images, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.show()

    def drawEllipseCov2D(self, ax, uv, cov2d, n_std=1.0, edgecolor='red', facecolor='none', linewidth=1, linestyle='-'):
        eigvals, eigvecs = np.linalg.eigh(cov2d)
        if np.any(eigvals <= 0):
            return
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigvals)

        ellipse = Ellipse(
            xy=uv,
            width=width,
            height=height,
            angle=angle,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
            linestyle=linestyle
        )
        ax.add_patch(ellipse)
        
    def project3D2D(self, sigma3D=None):
        if self.num_images == 0:
            return None, None
        # Project the 3D point to 2D coordinates using the projetion matrices
        intrinsic_matrices = np.stack([self.intrinsic_matrices[1] for _ in self.imageids])
        rotation_matrices = np.stack([self.rotation_matrices[imageid] for imageid in self.imageids])
        translation_vectors = np.stack([self.translation_vectors[imageid].reshape(3, 1) for imageid in self.imageids])
        point3D_world = self.point3Dcoord.copy().reshape(3, 1)
        point3D_camera = np.einsum('bij,jk->bik', rotation_matrices, point3D_world) + translation_vectors # (B, 3, 1)
        point3D_pixel = np.einsum('bij,bjk->bik', intrinsic_matrices, point3D_camera) # (B, 3, 1)
        point3D_uv, point3D_depth = point3D_pixel[:, :2, :] / point3D_pixel[:, 2:, :], point3D_pixel[:, 2:, :] # (B, 2, 1), (B, 1, 1)
        # Point projection done, now project 3D Gaussian with Jacobian of projection
        J = self.computeJacobian(point3D_camera, intrinsic_matrices, rotation_matrices)
        cov3Dmatrix = self.convertUpperTriangularCov3D(self.cov3Dmatrix.copy())
        if sigma3D is not None:
            cov3Dmatrix_proj = np.einsum('bij,jk,bkl->bil', J, sigma3D, J.transpose(0, 2, 1)) # (B, 2, 2)
            print("est")
        else:
            cov3Dmatrix_proj = np.einsum('bij,jk,bkl->bil', J, cov3Dmatrix, J.transpose(0, 2, 1))
        point3D_uv = np.round(point3D_uv).astype(int)
        return {"point3D_uv": point3D_uv, "cov3Dmatrix_proj": cov3Dmatrix_proj, "sigma3D": sigma3D, "point3D_depth": point3D_depth}

    def computeJacobian(self, camcoords, K, R):
        assert camcoords.shape[0] == K.shape[0] == R.shape[0]
        camcoords = camcoords.reshape(-1, 3)  # Ensure camcoords is 2D
        B = camcoords.shape[0]
        J = np.zeros((B, 2, 3))
        J[:, 0, 0] = K[:, 0, 0] / (camcoords[:, 2] ** 2)
        J[:, 1, 1] = K[:, 1, 1] / (camcoords[:, 2] ** 2)
        J[:, 0, 2] = -J[:, 0, 0] * camcoords[:, 0]
        J[:, 1, 2] = -J[:, 1, 1] * camcoords[:, 1]
        J = np.einsum('bij,bjk->bik', J, R)
        return J
    
    def convertUpperTriangularCov3D(self, cov3Dmatrix):
        """
        # Convert the covariance matrices from upper triangular form to full covariance matrices
        """
        cov3Dmatrix = cov3Dmatrix.reshape(-1)
        full_cov3Dmatrix = np.zeros((3, 3))
        full_cov3Dmatrix[0, 0] = cov3Dmatrix[0]
        full_cov3Dmatrix[0, 1] = cov3Dmatrix[1]
        full_cov3Dmatrix[0, 2] = cov3Dmatrix[2]
        full_cov3Dmatrix[1, 0] = cov3Dmatrix[1]
        full_cov3Dmatrix[1, 1] = cov3Dmatrix[3]
        full_cov3Dmatrix[1, 2] = cov3Dmatrix[4]
        full_cov3Dmatrix[2, 0] = cov3Dmatrix[2]
        full_cov3Dmatrix[2, 1] = cov3Dmatrix[4]
        full_cov3Dmatrix[2, 2] = cov3Dmatrix[5]
        return full_cov3Dmatrix
