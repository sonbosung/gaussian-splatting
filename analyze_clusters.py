import numpy as np
from scene.gaussian_model import GaussianModel
from utils.sh_utils import SH2RGB
from sklearn.cluster import DBSCAN
from collections import Counter

def main():
    ply_path = "/mnt/disk2/experiments/bicycle_init_test/point_cloud/iteration_0/point_cloud.ply"
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path)

    centers = gaussians._xyz.detach().cpu().numpy()
    colors = SH2RGB(gaussians._features_dc.squeeze(1).detach().cpu().numpy())
    
    # Spatial Clustering with DBSCAN
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    eps = 0.1  # This might need tuning based on the scene's scale
    min_samples = 5 # Minimum number of Gaussians to form a dense region
    
    print("Starting spatial clustering...")
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print(f'Estimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points: {n_noise_}')
    
    cluster_sizes = Counter(labels)
    
    # Analyze color similarity within each cluster
    print("\nAnalyzing color similarity within clusters...")
    for label in range(n_clusters_):
        cluster_points_indices = np.where(labels == label)[0]
        cluster_colors = colors[cluster_points_indices]
        
        # Calculate color standard deviation
        color_std_dev = np.std(cluster_colors, axis=0)
        avg_color_std_dev = np.mean(color_std_dev)
        
        print(f"Cluster {label}:")
        print(f"  - Number of primitives: {len(cluster_points_indices)}")
        print(f"  - Average color standard deviation: {avg_color_std_dev:.4f}")

if __name__ == "__main__":
    main()
