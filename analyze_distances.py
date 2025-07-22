import os
import sys
import numpy as np
import torch
from scene.gaussian_model import GaussianModel
from scipy.spatial import KDTree

def main():
    ply_path = "/mnt/disk2/exp/bicycle_newaug/point_cloud/iteration_0/point_cloud.ply"
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path)

    centers = gaussians._xyz.detach().cpu().numpy()
    num_gaussians = centers.shape[0]

    if num_gaussians < 2:
        print("Not enough Gaussians to compute distances.")
        return

    # Build a KD-tree for efficient nearest neighbor search
    tree = KDTree(centers)
    
    # Query for the 2 nearest neighbors; the first will be the point itself
    distances, _ = tree.query(centers, k=2)
    
    # The distances to the nearest neighbor are in the second column
    nearest_neighbor_distances = distances[:, 1]

    print(f"Number of Gaussians: {num_gaussians}")
    print(f"Average distance to nearest neighbor: {np.mean(nearest_neighbor_distances)}")
    print(f"Median distance to nearest neighbor: {np.median(nearest_neighbor_distances)}")
    print(f"Max distance to nearest neighbor: {np.max(nearest_neighbor_distances)}")
    print(f"Min distance to nearest neighbor: {np.min(nearest_neighbor_distances)}")


if __name__ == "__main__":
    main()
