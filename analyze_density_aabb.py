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

    tree = KDTree(centers)
    k = 5
    
    densities = np.zeros(num_gaussians)

    for i in range(num_gaussians):
        # Find k+1 nearest neighbors (including the point itself)
        distances, indices = tree.query(centers[i], k=k+1)
        
        # Get the coordinates of the neighbors
        knn_points = centers[indices]
        
        # Calculate the AABB volume
        min_coords = np.min(knn_points, axis=0)
        max_coords = np.max(knn_points, axis=0)
        volume = np.prod(max_coords - min_coords)
        
        # Calculate density
        if volume > 1e-9:  # Avoid division by zero or near-zero
            densities[i] = (k + 1) / volume
        else:
            densities[i] = np.inf # Assign infinite density if volume is zero

    print(f"Number of Gaussians: {num_gaussians}")
    print(f"Using k={k} for density estimation.")
    print(f"\nStatistics for density (k / AABB volume):")
    
    finite_densities = densities[np.isfinite(densities)]
    
    if finite_densities.size > 0:
        print(f"  - Average density: {np.mean(finite_densities)}")
        print(f"  - Median density: {np.median(finite_densities)}")
        print(f"  - Max density: {np.max(finite_densities)}")
        print(f"  - Min density: {np.min(finite_densities)}")
    else:
        print("All calculated densities were infinite.")

    num_infinite = np.isinf(densities).sum()
    print(f"  - Number of points with near-zero AABB volume (infinite density): {num_infinite}")

    # Identify the top 10 densest points
    # We consider finite densities first, then infinite ones if necessary
    sorted_indices = np.argsort(densities)[::-1] # Sort descending
    print("\nTop 10 densest points (indices and their density):")
    for i in range(min(10, num_gaussians)):
        idx = sorted_indices[i]
        print(f"  - Point index: {idx}, Density: {densities[idx]}")


if __name__ == "__main__":
    main()
