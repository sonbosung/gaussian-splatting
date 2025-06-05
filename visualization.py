import matplotlib.pyplot as plt
import numpy as np
from utils.scheduler_utils import ImageClustering
from utils.colmap_utils import compute_extrinsics
from sklearn.decomposition import PCA

def visualize_clustered_views(scene_path, cluster_num):
    scene = scene_path.split("/")[-1]
    dataset_path = scene_path + "/sparse/0"
    clustering = ImageClustering(dataset_path, cluster_num)
    train_images, points3D, cameras = clustering.train_images, clustering.points3D, clustering.cameras
    cam_center_dict = {}
    rotations_train_image, translations_train_image = compute_extrinsics(train_images)
    for i, key in enumerate(train_images.keys()):
        cam_center_dict[key]=(-rotations_train_image[key].T @ translations_train_image[key].reshape(3,1))

    cam_center = list(cam_center_dict.values())
    cam_center = np.array(cam_center)[:,:,0]
    pca = PCA(n_components=2)
    cam_center_2d = pca.fit_transform(cam_center)

    center_cam_center = np.mean(cam_center_2d, axis=0)

    ordered_names = [item for sublist in clustering.ordered_cluster_names.values() for item in sublist]
    names_to_key = {train_images[k].name: k for k in train_images.keys()}
    ordered_keys = [names_to_key[name] for name in ordered_names]
    plt.figure(figsize=(30,10))

    # Original ordering
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(cam_center_2d[:, 0], cam_center_2d[:, 1], 
                        c=np.arange(len(cam_center_2d)),
                        cmap='rainbow',
                        s=100, # Bigger dots
                        alpha=0.8)

    # Add sequence numbers
    for i, (x, y) in enumerate(cam_center_2d):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

    plt.title('Original Order')

    # Cluster visualization
    plt.subplot(1, 3, 2)
    cluster_ids = []
    for key in train_images.keys():
        name = train_images[key].name
        for cluster_id, names in clustering.ordered_cluster_names.items():
            if name in names:
                cluster_ids.append(cluster_id)
                break

    unique_clusters = sorted(set(cluster_ids))
    scatter_plots = []
    for cluster_id in unique_clusters:
        mask = np.array(cluster_ids) == cluster_id
        scatter = plt.scatter(cam_center_2d[mask, 0], cam_center_2d[mask, 1],
                            label=f'Cluster {cluster_id}',
                            s=100,
                            alpha=0.8)
        scatter_plots.append(scatter)

    plt.legend()
    plt.title('Clusters')

    # Ordered by flattened clusters
    plt.subplot(1, 3, 3)
    key_to_order = {k: i for i, k in enumerate(ordered_keys)}
    color_order = [key_to_order[key] for key in train_images.keys()]
    scatter = plt.scatter(cam_center_2d[:, 0], cam_center_2d[:, 1], 
                        c=color_order,
                        cmap='rainbow',
                        s=100, # Bigger dots 
                        alpha=0.8)

    # Add sequence numbers
    for i, (x, y) in enumerate(cam_center_2d):
        plt.annotate(str(color_order[i]), (x, y), xytext=(5, 5), textcoords='offset points')

    plt.title('Camera order after clustering')

    plt.tight_layout()
    plt.savefig(f"vis/{scene}_{cluster_num}_camera_order_visualization_cluster5.png")

from utils.scheduler_utils import GroupScheduler, PartialGroupScheduler
def visualize_scheduler_triggers(num_turns=None, scheduler_type="group", densify_from_iter=500, densify_until_iter=15000, n_cameras=200):
    # Create arrays to store trigger states
    iterations = np.arange(1, 30001)
    densify_triggers = np.zeros(30000)
    reset_opacity_triggers = np.zeros(30000)
    
    # Initialize scheduler
    if scheduler_type == "group":
        scheduler = GroupScheduler(None, None, densify_until_iter, densify_from_iter, debug=True)
    elif scheduler_type == "partial":
        scheduler = PartialGroupScheduler(None, None, densify_until_iter, densify_from_iter, debug=True)
    
    if num_turns:
        scheduler.set_num_turns(num_turns)

    # Run through iterations
    for iteration in range(1, 30001):
        scheduler.scheduled_training_index(iteration)
        if scheduler.densify_and_prune_flag:
            densify_triggers[iteration-1] = 1
            scheduler.densify_and_prune_flag = False
        if scheduler.reset_opacity_flag:
            reset_opacity_triggers[iteration-1] = 1
            scheduler.reset_opacity_flag = False
    
    # Create visualization
    plt.figure(figsize=(30, 10))
    
    # Plot densification triggers
    plt.subplot(2, 1, 1)
    plt.vlines(iterations[densify_triggers == 1], 0, 1, color='blue', label='Densify and Prune')
    plt.axvline(x=densify_from_iter, color='r', linestyle='--', label='Densify Start')
    plt.axvline(x=densify_until_iter, color='r', linestyle='--', label='Densify End')
    plt.title('Densification Triggers')
    plt.xlabel('Iteration')
    plt.ylabel('Trigger')
    plt.legend()
    plt.grid(True)
    
    # Plot reset opacity triggers
    plt.subplot(2, 1, 2)
    plt.vlines(iterations[reset_opacity_triggers == 1], 0, 1, color='green', label='Reset Opacity')
    plt.axvline(x=densify_from_iter, color='r', linestyle='--', label='Densify Start')
    plt.axvline(x=densify_until_iter, color='r', linestyle='--', label='Densify End')
    plt.title('Reset Opacity Triggers')
    plt.xlabel('Iteration')
    plt.ylabel('Trigger')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('scheduler_triggers_visualization.png')