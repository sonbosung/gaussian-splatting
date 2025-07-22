import numpy as np
from scene.gaussian_model import GaussianModel
from scipy.spatial import KDTree

def main():
    ply_path = "/mnt/disk2/exp/bicycle_newaug/point_cloud/iteration_0/point_cloud.ply"
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path)

    centers = gaussians._xyz.detach().cpu().numpy()
    num_gaussians = centers.shape[0]

    if num_gaussians < 2:
        print("Not enough Gaussians to compute densities.")
        return

    # Build a KD-tree for efficient nearest neighbor search
    tree = KDTree(centers)
    
    # k: number of nearest neighbors to consider for density estimation
    k = 5
    
    # Query for the k+1 nearest neighbors; the first will be the point itself
    # We get distances to the k nearest neighbors (excluding the point itself)
    distances, _ = tree.query(centers, k=k+1)
    
    # The distances to the k nearest neighbors are in columns 1 to k+1
    # We can use the distance to the k-th neighbor as a density estimate
    # or the average distance to the k neighbors. Let's use the average.
    avg_knn_distances = np.mean(distances[:, 1:], axis=1)

    print(f"Number of Gaussians: {num_gaussians}")
    print(f"Using k={k} for density estimation.")
    print(f"\nStatistics for average distance to {k} nearest neighbors (a measure of local density):")
    print(f"  - Average of average distances: {np.mean(avg_knn_distances)}")
    print(f"  - Median of average distances: {np.median(avg_knn_distances)}")
    print(f"  - Max of average distances (lowest density): {np.max(avg_knn_distances)}")
    print(f"  - Min of average distances (highest density): {np.min(avg_knn_distances)}")

    # Identify the top 10 densest points
    densest_indices = np.argsort(avg_knn_distances)[:10]
    print("\nTop 10 densest points (indices and their average k-NN distance):")
    for i in densest_indices:
        print(f"  - Point index: {i}, Avg k-NN distance: {avg_knn_distances[i]}")


if __name__ == "__main__":
    main()
