import os
import numpy as np
import torch
import cv2
from colmap.scripts.python.read_write_model import *
from tqdm import tqdm
import rtree
from shapely.geometry import Point, box
from collections import defaultdict
from utils.colmap_utils import compute_extrinsics, get_colmap_data, compute_intrinsics
from sklearn.neighbors import BallTree
from utils.scheduler_utils import ImageClustering
np.seterr(divide='ignore', invalid='ignore')

class Node:
    """A node in the Quadtree."""
    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.children = []
        self.sampled_point_uv = None
        self.sampled_point_rgb = None
        self.sampled_point_depth = None
        self.depth_interpolated = None
        self.neighbour_3D_indices = None
        self.matching_inference_count = 0
        self.matching_rejection_count = 0
        self.sampled_point_world = None
        self.unoccupied = True  # Indicates if the node is occupied by a sampled point
        self.matching_log = {}
        self.bounded_points3d_indices = []
        self.bounded_points3d_coords2d = []
        self.bounded_points3d_depths = []
        self.bounded_points3d_rgb = []
        self.sampled_point_neighbours_uv = None # Typo fixed: sampled_ponint_neighbours_uv -> sampled_point_neighbours_uv
        
    def get_error(self, img):
        """Compute the standard deviation of the region as an error metric."""
        region = img[self.y0:self.y0+self.height, self.x0:self.x0+self.width]
        return np.std(region) 

class Augmentor:
    """
    Augments the training data by sampling new 3D points in sparse regions of the scene.
    """
    def __init__(self,
                 image_dir,
                 colmap_dir,
                 mode="decompose",
                 quadtree_std_threshold=7,
                #  quadtree_min_pixel_length=5,
                 quadtree_min_pixel_length=10,
                 visibility_aware_culling=False,
                 n_clusters=20,
                 closest_n_views=12):
        self.image_dir = image_dir
        self.colmap_dir = colmap_dir
        self.mode = mode
        self.rgb_diff_threshold = 0.1
        self.depth_cutoff = 0.1
        self.cosine_threshold = 0.1
        self.quadtree_std_threshold = quadtree_std_threshold
        self.quadtree_min_pixel_length = quadtree_min_pixel_length
        self.visibility_aware_culling = visibility_aware_culling
        self.n_clusters = n_clusters
        self.closest_n_views_count = closest_n_views
        
        self._prepare_data()
        self.setup_image_clustering()
        self.construct_closest_n_views()
        self._augment()

    def setup_image_clustering(self):
        """Initializes image clustering."""
        self.image_clustering = ImageClustering(dataset_path=self.colmap_dir, n_clusters=self.n_clusters)

    def _prepare_data(self):
        """Loads and prepares COLMAP data."""
        self._load_colmap_data()
        self._compute_camera_parameters()
        self._gather_points3D()

    def _load_colmap_data(self):
        """Loads images, points, and cameras from COLMAP."""
        self.colmap_images, self.colmap_points3D, self.colmap_cameras = get_colmap_data(self.colmap_dir)
        self.split_train_test()
    
    def split_train_test(self):
        """Splits images into training and testing sets."""
        image_id_name = [[img.id, img.name] for img in self.colmap_images.values()]
        image_id_name_sorted = sorted(image_id_name, key=lambda x: x[1])
        self.train_ids = []
        self.test_ids = []
        for i, (image_id, image_name) in enumerate(image_id_name_sorted):
            if i % 8 == 0:
                self.test_ids.append(image_id)
            else:
                self.train_ids.append(image_id)
        self.image_keys = self.train_ids

    def _compute_camera_parameters(self):
        """Computes camera extrinsics and intrinsics."""
        self.rotations_image, self.translations_image = compute_extrinsics(self.colmap_images)
        image_sample = cv2.imread(os.path.join(self.image_dir,
                                               self.colmap_images[self.image_keys[0]].name))
        self.intrinsics_camera = compute_intrinsics(self.colmap_cameras,
                                                    image_sample.shape[1],
                                                    image_sample.shape[0])

    def _gather_points3D(self):
        """Gathers 3D points and their RGB values."""
        self.points3D = np.array([p.xyz for p in self.colmap_points3D.values()]).T
        self.points3D_rgb = np.array([p.rgb for p in self.colmap_points3D.values()]).T
        print(f"Total number of 3D points: {self.points3D.shape}")

    def quadtree_decomposition(self, image):
        """Decomposes an image into a quadtree."""
        height, width = image.shape[:2]
        root = Node(0, 0, width, height)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_norm = gray_image / 255.
        gauss = cv2.GaussianBlur(image_norm, (5, 5), 0)
        sobel_x = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=5)
        self.recursive_subdivide(root, gray_image)
        return root, sobel_x, sobel_y

    def recursive_subdivide(self, node, image):
        """Recursively subdivides a node in the quadtree."""
        if node.get_error(image) <= self.quadtree_std_threshold:
            return

        w_1 = node.width // 2
        w_2 = node.width - w_1
        h_1 = node.height // 2
        h_2 = node.height - h_1

        if w_1 < self.quadtree_min_pixel_length or h_1 < self.quadtree_min_pixel_length:
            return

        x1 = Node(node.x0, node.y0, w_1, h_1)
        self.recursive_subdivide(x1, image)
        x2 = Node(node.x0 + w_1, node.y0, w_2, h_1)
        self.recursive_subdivide(x2, image)
        y1 = Node(node.x0, node.y0 + h_1, w_1, h_2)
        self.recursive_subdivide(y1, image)
        y2 = Node(node.x0 + w_1, node.y0 + h_1, w_2, h_2)
        self.recursive_subdivide(y2, image)

        node.children = [x1, x2, y1, y2]

    def _augment(self):
        """Main augmentation loop."""
        self.roots = {}
        self.leaf_nodes = defaultdict(list)
        
        if self.mode == "decompose":
            for image_key in tqdm(self.image_keys, desc="Augmenting Images (decompose)"):
                self._process_image_for_decomposition(image_key)
        elif self.mode == "full":
            self.ix = {}
            self.iy = {}
            n_samples = 0
            for image_key in tqdm(self.image_keys, desc="Augmenting Images (full)"):
                n_samples += self._process_image_for_augmentation(image_key)
                tqdm.write(f"Processed image: {self.colmap_images[image_key].name} | Total points sampled: {n_samples}")
            self.correspondence_check()

    def _process_image_for_decomposition(self, image_key):
        """Processes a single image for quadtree decomposition."""
        image = cv2.imread(os.path.join(self.image_dir, self.colmap_images[image_key].name))
        root, _, _ = self.quadtree_decomposition(image)
        self.roots[image_key] = root
        self.collect_leaf_nodes(root, image_key)
        p3d_pix, p3d_depth = self.project_points3D_to_image(image_key)
        
        culled_indices = self._cull_points(image, p3d_pix, p3d_depth)
        self.check_points_in_leaf_node(image_key, p3d_pix, p3d_depth, culled_indices)

        for leaf_node in self.leaf_nodes[image_key]:
            leaf_node.sampled_point_uv = np.array([leaf_node.x0 + leaf_node.width // 2, leaf_node.y0 + leaf_node.height // 2])
            leaf_node.sampled_point_rgb = image[int(leaf_node.sampled_point_uv[1]), int(leaf_node.sampled_point_uv[0])]

    def _process_image_for_augmentation(self, image_key):
        """Processes a single image for full augmentation."""
        image = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, self.colmap_images[image_key].name)), cv2.COLOR_BGR2RGB)
        root, ix, iy = self.quadtree_decomposition(image)
        self.roots[image_key] = root
        self.ix[image_key] = ix
        self.iy[image_key] = iy
        self.collect_leaf_nodes(root, image_key)
        
        p3d_pix, p3d_depth = self.project_points3D_to_image(image_key)
        culled_indices = self._cull_points(image, p3d_pix, p3d_depth)
        self.check_points_in_leaf_node(image_key, p3d_pix, p3d_depth, culled_indices)

        for leaf_node in self.leaf_nodes[image_key]:
            if leaf_node.unoccupied:
                leaf_node.sampled_point_uv = np.array([leaf_node.x0 + leaf_node.width // 2, leaf_node.y0 + leaf_node.height // 2])
                leaf_node.sampled_point_rgb = image[int(leaf_node.sampled_point_uv[1]), int(leaf_node.sampled_point_uv[0])]
        
        return self.find_depth_from_nearest_neighbors(image_key, p3d_pix, p3d_depth, culled_indices)

    def _cull_points(self, image, p3d_pix, p3d_depth):
        """Performs frustum and visibility culling."""
        near_culled_indices = self.camera_frustum_culling_pix(image.shape[1], image.shape[0], p3d_pix, p3d_depth)
        if self.visibility_aware_culling:
            visibility_culled_indices = self.visibility_culling(image, p3d_pix, near_culled_indices)
            return near_culled_indices | visibility_culled_indices
        return near_culled_indices

    def find_depth_from_nearest_neighbors(self, image_key, pix_coords, depths, culled_indices):
        """Finds depth for sampled points using nearest neighbors."""
        sampled_points = np.array([leaf.sampled_point_uv for leaf in self.leaf_nodes[image_key] if leaf.unoccupied]).T
        if sampled_points.shape[0] == 0:
            return 0

        original_indices = np.arange(pix_coords.shape[1])[~culled_indices]
        pix_coords = pix_coords[:, ~culled_indices]
        depths = depths[:, ~culled_indices]

        tree = BallTree(pix_coords.T, leaf_size=40)
        _, indices = tree.query(sampled_points.T, k=3)

        inverse_intrinsics = np.linalg.inv(self.intrinsics_camera[self.colmap_images[image_key].camera_id])
        
        sampled_points_h = np.vstack((sampled_points, np.ones((1, sampled_points.shape[1]))))
        sampled_points_cameraframe = inverse_intrinsics @ sampled_points_h
        
        pix_coords_h = np.vstack((pix_coords, np.ones((1, pix_coords.shape[1]))))
        pix_coords_cameraframe = inverse_intrinsics @ pix_coords_h
        
        a, b, c, d = self.compute_normal_vector(pix_coords_cameraframe, depths, indices)
        sampled_points_depth, cosine_culled_indices = self.compute_depth(sampled_points_cameraframe, a, b, c, d)
        
        depth_rejected_indices = (sampled_points_depth < self.depth_cutoff).reshape(-1) | \
                                  np.isnan(sampled_points_cameraframe).any(axis=0).reshape(-1) | \
                                  cosine_culled_indices.reshape(-1)
                                  
        sampled_points_world = self.transform_camera_to_world(sampled_points_cameraframe, sampled_points_depth, image_key)

        augmented_count = 0
        depth_index = 0
        for leaf_node in self.leaf_nodes[image_key]:
            if leaf_node.unoccupied:
                if not depth_rejected_indices[depth_index]:
                    leaf_node.sampled_point_depth = sampled_points_depth[:, depth_index]
                    leaf_node.sampled_point_world = sampled_points_world[:, depth_index]
                    leaf_node.depth_interpolated = True
                    leaf_node.sampled_point_neighbours_indices = original_indices[indices[depth_index]]
                    leaf_node.sampled_point_neighbours_uv = pix_coords[:, indices[depth_index]]
                    augmented_count += 1
                else:
                    leaf_node.depth_interpolated = False
                depth_index += 1
        return augmented_count
    
    def transform_camera_to_world(self, sampled_points_cameraframe, sampled_points_depth, image_key):
        """Transforms points from camera frame to world frame."""
        scaled_points = sampled_points_cameraframe[:2,:] * sampled_points_depth.reshape(1,-1)
        points_in_camera = np.concatenate((scaled_points, sampled_points_depth.reshape(1,-1)), axis=0)
        
        R = self.rotations_image[image_key]
        t = self.translations_image[image_key]
        
        # Create 4x4 transformation matrix from camera to world
        Rt = np.concatenate((R, t.reshape(3,1)), axis=1)
        world_to_camera_matrix = np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)
        camera_to_world_matrix = np.linalg.inv(world_to_camera_matrix)

        points_in_camera_h = np.vstack((points_in_camera, np.ones((1, points_in_camera.shape[1]))))
        points_in_world = camera_to_world_matrix @ points_in_camera_h
        
        return points_in_world[:3,:]

    def compute_depth(self, sampled_points_cameraframe, a, b, c, d):
        """Computes depth of sampled points."""
        direction_vectors = sampled_points_cameraframe
        normal_vectors = np.concatenate((a.reshape(1,-1), b.reshape(1,-1), c.reshape(1,-1)), axis=0)
        
        numerator = -d.reshape(1,-1)
        denominator = np.sum(normal_vectors * direction_vectors, axis=0).reshape(1,-1)
        
        # Avoid division by zero
        denominator[denominator == 0] = 1e-6
        t = numerator / denominator

        cosine_similarity = np.abs(np.sum(normal_vectors * direction_vectors, axis=0)) / \
                            (np.linalg.norm(normal_vectors, axis=0) * np.linalg.norm(direction_vectors, axis=0))
        cosine_culled_indices = cosine_similarity < self.cosine_threshold
        
        depth = t * direction_vectors[2,:]
        return depth.reshape(1, -1), cosine_culled_indices
        
    def compute_normal_vector(self, pix_coords_cameraframe, depths, indices):
        """Computes normal vectors for planes defined by 3 points."""
        points_in_camera = np.concatenate((pix_coords_cameraframe[:2,:] * depths.reshape(1,-1),
                                           depths.reshape(1,-1)), axis=0)
        p1 = points_in_camera[:, indices[:, 0]]
        p2 = points_in_camera[:, indices[:, 1]]
        p3 = points_in_camera[:, indices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        
        normals = np.cross(v1.T, v2.T).T
        a, b, c = normals[0], normals[1], normals[2]
        d = -np.sum(normals * p1, axis=0)
        return a, b, c, d

    def check_points_in_leaf_node(self, image_key, pix_coords, depths, culled_indices):
        """Checks which 3D points fall into which leaf nodes."""
        rect_index = rtree.index.Index()
        leaf_nodes = self.leaf_nodes[image_key]
        rectangles = [((node.x0, node.y0, node.x0 + node.width, node.y0 + node.height), i) for i, node in enumerate(leaf_nodes)]
        for i, rect_tuple in enumerate(rectangles):
            rect_index.insert(i, rect_tuple[0])

        for i, (x, y) in enumerate(pix_coords.T):
            if culled_indices[i]:
                continue
            
            matches = list(rect_index.intersection((x, y, x, y)))
            for match_idx in matches:
                node_rect = rectangles[match_idx][0]
                if box(*node_rect).contains(Point(x, y)):
                    leaf_nodes[match_idx].unoccupied = False
                    leaf_nodes[match_idx].bounded_points3d_indices.append(i)
                    leaf_nodes[match_idx].bounded_points3d_depths.append(depths[0, i])
                    leaf_nodes[match_idx].bounded_points3d_rgb.append(self.points3D_rgb[:, i])

    def collect_leaf_nodes(self, node, image_key):
        """Collects all leaf nodes from the quadtree."""
        if not node.children:
            self.leaf_nodes[image_key].append(node)
        else:
            for child in node.children:
                self.collect_leaf_nodes(child, image_key)

    def project_points3D_to_image(self, image_key):
        """Projects 3D points to the image plane."""
        K = self.intrinsics_camera[self.colmap_images[image_key].camera_id]
        R = self.rotations_image[image_key]
        t = self.translations_image[image_key]
        P = K @ np.concatenate((R, t.reshape(3,1)), axis=1)
        
        p3D_h = np.vstack((self.points3D, np.ones((1, self.points3D.shape[1]))))
        p3D_pix = P @ p3D_h
        
        p3D_depth = p3D_pix[2:3]
        # Avoid division by zero for depths
        p3D_depth[p3D_depth == 0] = 1e-6
        p3D_pix = p3D_pix[:2] / p3D_depth
        
        return p3D_pix, p3D_depth
    
    def project_world_to_image(self, image_key, coords3D, width, height):
        """Projects 3D world coordinates to the image plane."""
        K = self.intrinsics_camera[self.colmap_images[image_key].camera_id]
        R = self.rotations_image[image_key]
        t = self.translations_image[image_key]
        P = K @ np.concatenate((R, t.reshape(3,1)), axis=1)
        
        coords3D_h = np.vstack((coords3D, np.ones((1, coords3D.shape[1]))))
        coords2D_h = P @ coords3D_h
        
        depths = coords2D_h[2:3]
        # Avoid division by zero for depths
        depths[depths == 0] = 1e-6
        coords2D = coords2D_h[:2] / depths
        
        culled_indices = self.camera_frustum_culling_pix(width, height, coords2D, depths)
        return coords2D, depths, culled_indices
    
    def camera_frustum_culling_pix(self, width, height, pix_coords, depths):
        """Performs camera frustum culling in pixel coordinates."""
        culled_indices = (pix_coords[0] < 0) | \
                         (pix_coords[0] >= width) | \
                         (pix_coords[1] < 0) | \
                         (pix_coords[1] >= height) | \
                         (depths < 0).reshape(-1) | \
                         np.isnan(pix_coords).any(axis=0)
        return culled_indices
    
    def visibility_culling(self, image, pix_coords, near_culled_indices):
        """Performs visibility culling based on RGB difference."""
        visibility_culled_indices = np.zeros_like(near_culled_indices, dtype=bool)
        for i, pix_coord in enumerate(pix_coords.T):
            if near_culled_indices[i]:
                continue
            image_point_rgb = image[int(pix_coord[1]), int(pix_coord[0])]
            colmap_point_rgb = self.points3D_rgb[:, i]
            rgb_diff = self.pixelwise_rgb_diff(image_point_rgb, colmap_point_rgb)
            visibility_culled_indices[i] = rgb_diff < self.rgb_diff_threshold
        return visibility_culled_indices
    
    def pixelwise_rgb_diff(self, image_rgb, colmap_rgb):
        """Computes the L2 norm of the RGB difference."""
        def normalize(rgb):
            if np.max(rgb) > 1.0:
                return rgb / 255.0
            return rgb
        
        image_rgb_norm = normalize(image_rgb)
        colmap_rgb_norm = normalize(colmap_rgb)
        return np.linalg.norm(image_rgb_norm - colmap_rgb_norm)

    def find_corresponding_leaf_node(self, image_key, pix_coord, scale=1.0):
        """Finds the leaf node corresponding to a pixel coordinate."""
        pix_coord = np.array(pix_coord) / scale
        root = self.roots[image_key]
        return self._find_leaf_recursive(root, pix_coord)

    def _find_leaf_recursive(self, node, pix_coord):
        """Recursively finds the leaf node containing the pixel coordinate."""
        if not node.children:
            return node
        for child in node.children:
            if child.x0 <= pix_coord[0] < child.x0 + child.width and \
               child.y0 <= pix_coord[1] < child.y0 + child.height:
                return self._find_leaf_recursive(child, pix_coord)
        return None

    def construct_closest_n_views(self):
        """Constructs a list of N closest views for each view."""
        self.closest_n_views = {}
        self.name_to_key = {self.colmap_images[key].name: key for key in self.image_keys}
        for view in self.image_keys:
            view_name = self.colmap_images[view].name
            similarities = self.image_clustering.W_dict[view_name].items()
            # Exclude self-similarity and sort
            sorted_similarities = sorted([s for s in similarities if s[0] != view_name], key=lambda x: x[1], reverse=True)
            # Get top N closest views
            closest_n_names = [name for name, _ in sorted_similarities[:self.closest_n_views_count]]
            self.closest_n_views[view] = [self.name_to_key[name] for name in closest_n_names]
    
    def visualize_cliques(self, cliques, num_cliques_to_viz=5):
        """Visualizes a few cliques by saving annotated images."""
        debug_dir = os.path.join(os.path.dirname(self.image_dir), "debug_cliques")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"DEBUG: Saving clique visualizations to {debug_dir}")

        # To avoid visualizing too many, especially if cliques are large
        cliques_to_viz = cliques[:num_cliques_to_viz]

        for i, clique in enumerate(cliques_to_viz):
            clique_img_dir = os.path.join(debug_dir, f"clique_{i}")
            os.makedirs(clique_img_dir, exist_ok=True)
            
            for view, node_idx in clique:
                image_info = self.colmap_images[view]
                image_path = os.path.join(self.image_dir, image_info.name)
                image = cv2.imread(image_path)
                
                leaf_node = self.leaf_nodes[view][node_idx]
                uv = leaf_node.sampled_point_uv
                
                # Draw a circle at the sampled point
                cv2.circle(image, (int(uv[0]), int(uv[1])), 10, (0, 0, 255), 2) # Red circle
                
                # Draw neighbor points if available
                if hasattr(leaf_node, 'sampled_point_neighbours_uv') and leaf_node.sampled_point_neighbours_uv is not None:
                    for neighbor_uv in leaf_node.sampled_point_neighbours_uv.T:
                        cv2.circle(image, (int(neighbor_uv[0]), int(neighbor_uv[1])), 5, (255, 0, 0), -1) # Blue dots for neighbours

                # Save the image
                save_path = os.path.join(clique_img_dir, f"view_{view}_{image_info.name}")
                cv2.imwrite(save_path, image)

    def correspondence_check(self):
        """Checks for correspondences between sampled points across different views."""
        bidirectional_edges = self._find_bidirectional_correspondences()
        self.bidirectional_edges = bidirectional_edges
        
        maximal_cliques = self._find_maximal_cliques(bidirectional_edges)
        self.merged_correspondences = maximal_cliques
        
        # Debugging: Print clique stats and visualize
        if maximal_cliques:
            clique_sizes = [len(c) for c in maximal_cliques]
            print(f"DEBUG: Found {len(maximal_cliques)} maximal cliques.")
            print(f"DEBUG: Clique size distribution: Min={min(clique_sizes)}, Max={max(clique_sizes)}, Avg={np.mean(clique_sizes):.2f}")
            self.visualize_cliques(maximal_cliques)
        else:
            print("DEBUG: No maximal cliques found.")
        
        self._merge_cliques()
        
        if hasattr(self, 'merged_sample_points_world') and self.merged_sample_points_world:
            print(f"Total number of merged sampled points: {len(self.merged_sample_points_world)}")
            print(f"Total number of cliques: {len(self.merged_correspondences)}")
        else:
            print("No points were merged.")

    def _find_bidirectional_correspondences(self):
        """Finds bidirectional correspondences between leaf nodes."""
        all_edges = []
        for view in tqdm(self.image_keys, desc="Finding Correspondences"):
            # Gather all valid sampled points from the current view
            sampled_points_data = []
            for i, leaf_node in enumerate(self.leaf_nodes[view]):
                if leaf_node.depth_interpolated and leaf_node.sampled_point_world is not None and not np.isnan(leaf_node.sampled_point_world).any():
                    sampled_points_data.append({
                        "world_coords": leaf_node.sampled_point_world,
                        "rgb": leaf_node.sampled_point_rgb,
                        "node_idx": i
                    })
            
            if not sampled_points_data:
                continue

            sampled_points_world = np.array([d["world_coords"] for d in sampled_points_data]).T
            sampled_points_rgb = np.array([d["rgb"] for d in sampled_points_data])
            target_node_indices = np.array([d["node_idx"] for d in sampled_points_data])

            for ref_view in self.closest_n_views[view]:
                # Project points to reference view
                coords2d_ref, depth_ref, culled = self.project_world_to_image(ref_view, sampled_points_world, self.roots[ref_view].width, self.roots[ref_view].height)
                
                # Filter culled points
                valid_indices = ~culled
                coords2d_ref = coords2d_ref[:, valid_indices]
                depth_ref = depth_ref[:, valid_indices]
                rgb_ref = sampled_points_rgb[valid_indices]
                target_node_indices_ref = target_node_indices[valid_indices]

                for i, point_2d in enumerate(coords2d_ref.T):
                    ref_leaf_node = self.find_corresponding_leaf_node(ref_view, point_2d)
                    if ref_leaf_node and ref_leaf_node.depth_interpolated:
                        # Check for correspondence
                        depth_error = np.abs(ref_leaf_node.sampled_point_depth - depth_ref[0, i])
                        rgb_diff = self.pixelwise_rgb_diff(ref_leaf_node.sampled_point_rgb, rgb_ref[i])
                        
                        # Dynamic depth bound based on leaf node size
                        depth_bound = ref_leaf_node.sampled_point_depth * np.tan(np.deg2rad(0.5))

                        if depth_error < depth_bound and rgb_diff < self.rgb_diff_threshold:
                            target_node_idx = target_node_indices_ref[i]
                            ref_node_idx = self.leaf_nodes[ref_view].index(ref_leaf_node)
                            all_edges.append(tuple(sorted(((view, target_node_idx), (ref_view, ref_node_idx)))))

        # Find bidirectional edges by counting occurrences
        edge_counts = defaultdict(int)
        for edge in all_edges:
            edge_counts[edge] += 1
        
        bidirectional_edges = [list(edge[0]) + list(edge[1]) for edge, count in edge_counts.items() if count > 1]
        print(f"DEBUG: Found {len(all_edges)} total edges, resulting in {len(bidirectional_edges)} bidirectional edges.")
        return bidirectional_edges

    def _find_maximal_cliques(self, edges):
        """Finds maximal cliques in the graph of correspondences."""
        graph = defaultdict(set)
        all_nodes = set()
        for v1, n1, v2, n2 in edges:
            node1 = (v1, n1)
            node2 = (v2, n2)
            graph[node1].add(node2)
            graph[node2].add(node1)
            all_nodes.add(node1)
            all_nodes.add(node2)

        def bron_kerbosch(R, P, X, cliques):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            
            if not P:
                return

            pivot = next(iter(P.union(X)))
            P_without_neighbors_of_pivot = P.difference(graph[pivot])
            
            for node in list(P_without_neighbors_of_pivot):
                neighbors = graph[node]
                bron_kerbosch(R.union({node}), P.intersection(neighbors), X.intersection(neighbors), cliques)
                P.remove(node)
                X.add(node)

        cliques = []
        bron_kerbosch(set(), all_nodes, set(), cliques)
        return cliques

    def _merge_cliques(self):
        """Merges the points within each clique."""
        self.merged_sample_points_world = []
        self.merged_sample_points_rgb = []
        self.merged_sample_points_indices = []
        self.merged_image_ids = []
        self.mean_labels = []
        self.cov_labels = []
        self.rgb_labels = []

        for i, group in enumerate(self.merged_correspondences):
            world_points = [self.leaf_nodes[v][n].sampled_point_world for v, n in group]
            rgb_values = [self.leaf_nodes[v][n].sampled_point_rgb for v, n in group]

            self.merged_sample_points_world.append(np.mean(world_points, axis=0))
            self.merged_sample_points_rgb.append(np.mean(rgb_values, axis=0))

            for v, n in group:
                leaf = self.leaf_nodes[v][n]
                mean = leaf.sampled_point_uv
                ix = self.ix[v][mean[1], mean[0]]
                iy = self.iy[v][mean[1], mean[0]]
                
                cov = self._calculate_covariance(ix, iy, leaf.width, leaf.height)

                self.merged_image_ids.append(v)
                self.merged_sample_points_indices.append(i)
                self.mean_labels.append(mean)
                self.cov_labels.append(cov)
                self.rgb_labels.append(leaf.sampled_point_rgb)

    def _calculate_covariance(self, ix, iy, width, height):
        """Calculates the 2D covariance for a sampled point."""
        w_sq_36 = width**2 / 36
        h_sq_36 = height**2 / 36

        if abs(ix) < 1e-3 and abs(iy) < 1e-3:
            return np.array([[w_sq_36, 0], [0, h_sq_36]])
        
        structure_tensor = np.array([[ix*ix, ix*iy], [ix*iy, iy*iy]])
        eigvals, eigvecs = np.linalg.eigh(structure_tensor)
        
        eigvals_max = np.max(eigvals)
        if eigvals_max > 0:
            eigvals = np.maximum(eigvals / eigvals_max, 0.33)
        
        eigvals = np.sqrt(eigvals) * width / 6
        scale_matrix = np.diag([eigvals[1], eigvals[0]])
        
        return eigvecs @ scale_matrix @ scale_matrix @ eigvecs.T