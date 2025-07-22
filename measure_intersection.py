import os
import sys
import numpy as np
import torch
from scene.gaussian_model import GaussianModel
from utils.sh_utils import SH2RGB
import pdb
from scipy.spatial import KDTree

def recover_full(L):
    """Recover the full covariance matrix from the upper triangular matrix L."""
    full_covariance = np.zeros((L.shape[0], 3, 3))
    full_covariance[:, 0, 0] = L[:, 0]
    full_covariance[:, 0, 1] = L[:, 1]
    full_covariance[:, 1, 0] = L[:, 1]
    full_covariance[:, 0, 2] = L[:, 2]
    full_covariance[:, 2, 0] = L[:, 2]
    full_covariance[:, 1, 1] = L[:, 3]
    full_covariance[:, 1, 2] = L[:, 4]
    full_covariance[:, 2, 1] = L[:, 4]
    full_covariance[:, 2, 2] = L[:, 5]
    return full_covariance

def main():
    ply_path = "/mnt/disk2/experiments/bicycle_init_pure/point_cloud/iteration_0/point_cloud.ply"
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path)

    centers = gaussians._xyz.detach().cpu().numpy()
    cov_mat = gaussians.get_covariance().detach().cpu().numpy()
    covariances = recover_full(cov_mat)

    num_gaussians = centers.shape[0]
    intersection_scores = []

    # Build a KD-tree for efficient nearest neighbor search
    tree = KDTree(centers)
    
    # For each Gaussian, find its k nearest neighbors and compute intersection
    k = 10  # Number of nearest neighbors to consider
    for i in range(num_gaussians):
        distances, indices = tree.query(centers[i], k=k+1) # k+1 because the point itself is included
        
        for j_idx in range(1, len(indices)): # Start from 1 to exclude the point itself
            j = indices[j_idx]
            if i >= j: # Avoid double counting and self-comparison
                continue

            mu1 = centers[i]
            cov1 = covariances[i]
            mu2 = centers[j]
            cov2 = covariances[j]

            # Bhattacharyya distance for two multivariate normal distributions
            try:
                cov_mean = (cov1 + cov2) / 2
                cov_mean_inv = np.linalg.inv(cov_mean)
                
                term1 = 0.125 * (mu2 - mu1).T @ cov_mean_inv @ (mu2 - mu1)
                
                det_cov1 = np.linalg.det(cov1)
                det_cov2 = np.linalg.det(cov2)
                det_cov_mean = np.linalg.det(cov_mean)

                # Add a small epsilon to avoid log(0) or division by zero
                epsilon = 1e-10
                denominator = np.sqrt(np.abs(det_cov1 * det_cov2) + epsilon)
                if denominator == 0:
                    continue # or handle as a special case

                term2 = 0.5 * np.log(det_cov_mean / denominator + epsilon)
                
                bhattacharyya_distance = term1 + term2
                
                # Convert distance to a similarity score (0 to 1, where 1 is high intersection)
                intersection_score = np.exp(-bhattacharyya_distance)
                intersection_scores.append(intersection_score)
            except np.linalg.LinAlgError:
                # This can happen if a covariance matrix is singular
                continue

    print(f"Number of Gaussians: {num_gaussians}")
    if intersection_scores:
        print(f"Number of pairs considered: {len(intersection_scores)}")
        print(f"Average intersection score: {np.mean(intersection_scores)}")
        print(f"Median intersection score: {np.median(intersection_scores)}")
        print(f"Max intersection score: {np.max(intersection_scores)}")
        print(f"Min intersection score: {np.min(intersection_scores)}")
    else:
        print("No intersections found or all covariance matrices were singular.")

if __name__ == "__main__":
    main()
