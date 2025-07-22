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
        
        for i in range(num_images):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            uv_main = uvs[i]
            cov_main = proj_cov2Ds[i]
            uv_ref = self.point2Dcoords[i]
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
            x1, x2 = int(x0 - zoom_size), int(x0 + zoom_size)
            y1, y2 = int(y0 - zoom_size), int(y0 + zoom_size)

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

    def drawEllipseCov2D(self, ax, uv, cov2d, n_std=3.0, edgecolor='red', facecolor='none', linewidth=1, linestyle='-'):
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
        
    def project3D2D(self):
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
        cov3Dmatrix_proj = np.einsum('bij,jk,bkl->bil', J, cov3Dmatrix, J.transpose(0, 2, 1)) # (B, 2, 2)
        point3D_uv = np.round(point3D_uv).astype(int)
        return point3D_uv, cov3Dmatrix_proj
        
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
