import os
import numpy as np
import torch
import cv2
from sklearn.neighbors import NearestNeighbors
from colmap.scripts.python.read_write_model import *
from tqdm import tqdm
from sklearn.decomposition import PCA
import rtree
from shapely.geometry import Point, box
from collections import defaultdict
from utils.colmap_utils import compute_extrinsics, get_colmap_data, compute_intrinsics
from matplotlib import pyplot as plt
from sklearn.neighbors import BallTree
from utils.scheduler_utils import ImageClustering
np.seterr(divide='ignore', invalid='ignore')

class Node:
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
        self.sampled_ponint_neighbours_uv = None
        
    def get_error(self, img):
        # Compute the standard deviation of the region as an error metric
        region = img[self.y0:self.y0+self.height, self.x0:self.x0+self.width]
        return np.std(region) 

class Augmentor:
    def __init__(self,
                 image_dir,
                 colmap_dir,
                 mode="decompose",
                 quadtree_std_threshold=7,
                 quadtree_min_pixel_length=5,
                 visibility_aware_culling=False):
        self.image_dir = image_dir
        self.colmap_dir = colmap_dir
        self.mode = mode
        self.rgb_diff_threshold = 0.1
        self.depth_cutoff = 0.1
        self.cosine_threshold = 0.1
        self.quadtree_std_threshold = quadtree_std_threshold
        self.quadtree_min_pixel_length = quadtree_min_pixel_length
        self.visibility_aware_culling = visibility_aware_culling
        
        self._prepare_data()
        self.setup_image_clustering()
        self.construct_closest_n_views(n=12)
        self._augment()

    def setup_image_clustering(self):
        self.image_clustering = ImageClustering(dataset_path=self.colmap_dir, n_clusters=20)

    def _prepare_data(self):
        self._load_colmap_data()
        self._compute_camera_parameters()
        self._gather_points3D()

    def _load_colmap_data(self):
        self.colmap_images, self.colmap_points3D, self.colmap_cameras = get_colmap_data(self.colmap_dir)
        # self.image_keys = list(self.colmap_images.keys())
        self.split_train_test()
    
    def split_train_test(self):
        image_id_name = [[self.colmap_images[key].id, self.colmap_images[key].name] for key in self.colmap_images.keys()]
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
        self.rotations_image, self.translations_image = compute_extrinsics(self.colmap_images)
        image_sample = cv2.imread(os.path.join(self.image_dir,
                                               self.colmap_images[self.image_keys[0]].name))
        self.intrinsics_camera = compute_intrinsics(self.colmap_cameras,
                                                    image_sample.shape[1],
                                                    image_sample.shape[0])

    def _gather_points3D(self):
        # Each points are column vectors!
        self.points3D = []
        self.points3D_rgb = []
        self.points3D_depth = []
        for point3D in self.colmap_points3D.values():
            self.points3D.append(point3D.xyz)
            self.points3D_rgb.append(point3D.rgb)
        self.points3D = np.array(self.points3D).T
        self.points3D_rgb = np.array(self.points3D_rgb).T
        print(f"Total number of 3D points: {self.points3D.shape}")

    def quadtree_decomposition(self, image):
        height, width = image.shape[:2]
        root = Node(0, 0, width, height)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_norm = image / 255.
        gauss = cv2.GaussianBlur(image_norm, (5, 5), 0)
        sobel_x = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=5)
        self.recursive_subdivide(root, image)
        return root, sobel_x, sobel_y

    def recursive_subdivide(self, node, image):
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
        self.roots = {}
        self.leaf_nodes = defaultdict(list)
        n_samples = 0
        if self.mode == "decompose":
            for image_key in tqdm(self.image_keys, desc=f"Augmenting Images, {n_samples} points sampled"):
                image = cv2.imread(os.path.join(self.image_dir, self.colmap_images[image_key].name))
                root, _, _ = self.quadtree_decomposition(image)
                self.roots[image_key] = root
                self.collect_leaf_nodes(self.roots[image_key], image_key)
                p3d_pix, p3d_depth = self.project_points3D_to_image(image_key)
                near_culled_indices = self.camera_frustum_culling_pix(image.shape[1], 
                                                                    image.shape[0], 
                                                                    p3d_pix, 
                                                                    p3d_depth)
                if self.visibility_aware_culling:
                    visibility_culled_indices = self.visibility_culling(image, p3d_pix, near_culled_indices)
                    near_culled_indices = near_culled_indices | visibility_culled_indices
                # print(f"near_culled_indices.shape: {near_culled_indices.shape}")
                self.check_points_in_leaf_node(image_key, p3d_pix, p3d_depth, near_culled_indices)

                sampled_points = []
                sampled_points_rgb = []
                for leaf_node in self.leaf_nodes[image_key]:
                    leaf_node.sampled_point_uv = np.array([leaf_node.x0+leaf_node.width//2, leaf_node.y0+leaf_node.height//2])
                    leaf_node.sampled_point_rgb = image[int(leaf_node.sampled_point_uv[1]),
                                                        int(leaf_node.sampled_point_uv[0])]
 
        elif self.mode == "full":
            self.ix = {}
            self.iy = {}
            for image_key in tqdm(self.image_keys, desc=f"Augmenting Images, {n_samples} points sampled"):
                tqdm.write(f"Processing image: {self.colmap_images[image_key].name} | {n_samples} points sampled")

                image = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, self.colmap_images[image_key].name)), cv2.COLOR_BGR2RGB)
                root, ix, iy = self.quadtree_decomposition(image)
                self.roots[image_key] = root
                self.ix[image_key] = ix
                self.iy[image_key] = iy
                self.collect_leaf_nodes(self.roots[image_key], image_key)
                p3d_pix, p3d_depth = self.project_points3D_to_image(image_key)
                near_culled_indices = self.camera_frustum_culling_pix(image.shape[1], 
                                                                    image.shape[0], 
                                                                    p3d_pix, 
                                                                    p3d_depth)
                if self.visibility_aware_culling:
                    visibility_culled_indices = self.visibility_culling(image, p3d_pix, near_culled_indices)
                    near_culled_indices = near_culled_indices | visibility_culled_indices
                # print(f"near_culled_indices.shape: {near_culled_indices.shape}")
                self.check_points_in_leaf_node(image_key, p3d_pix, p3d_depth, near_culled_indices)

                sampled_points = []
                sampled_points_rgb = []
                for leaf_node in self.leaf_nodes[image_key]:
                    if leaf_node.unoccupied:
                        leaf_node.sampled_point_uv = np.array([leaf_node.x0+leaf_node.width//2, leaf_node.y0+leaf_node.height//2])
                        leaf_node.sampled_point_rgb = image[int(leaf_node.sampled_point_uv[1]),
                                                            int(leaf_node.sampled_point_uv[0])]
                        sampled_points.append(leaf_node.sampled_point_uv)
                        sampled_points_rgb.append(leaf_node.sampled_point_rgb)
                sampled_points = np.array(sampled_points).T
                sampled_points_rgb = np.array(sampled_points_rgb)
                augmented_count = self.find_depth_from_nearest_neighbors(image_key,
                                                    p3d_pix,
                                                    p3d_depth,
                                                    near_culled_indices,)
                n_samples += augmented_count
            self.correspondence_check()
            
            
    def find_depth_from_nearest_neighbors(self, image_key, pix_coords, depths, near_culled_indices):
        sampled_points = []
        for leaf_node in self.leaf_nodes[image_key]:
            if leaf_node.unoccupied:
                sampled_points.append(leaf_node.sampled_point_uv)

        sampled_points = np.array(sampled_points).T

        original_indices = np.arange(pix_coords.shape[1])
        original_indices = original_indices[~near_culled_indices[0]]

        pix_coords = pix_coords[:,~near_culled_indices]
        depths = depths[:,~near_culled_indices]

        tree = BallTree(pix_coords.T, leaf_size=40)
        distances, indices = tree.query(sampled_points.T, k=3)

        inverse_intrinsics = np.linalg.inv(self.intrinsics_camera[self.colmap_images[image_key].camera_id])
        sampled_points_h = np.vstack((sampled_points, np.ones((1, sampled_points.shape[1]))))
        sampled_points_cameraframe = inverse_intrinsics @ sampled_points_h
        pix_coords_h = np.vstack((pix_coords, np.ones((1, pix_coords.shape[1]))))
        pix_coords_cameraframe = inverse_intrinsics @ pix_coords_h
        a, b, c, d = self.compute_normal_vector(pix_coords_cameraframe, depths, indices)
        sampled_points_depth, cosine_culled_indices = self.compute_depth(sampled_points_cameraframe,
                                                                         a, b, c, d)
        depth_rejected_indices = (sampled_points_depth < self.depth_cutoff).reshape(-1) |\
                                  np.isnan(sampled_points_cameraframe).any(axis=0).reshape(-1) |\
                                  cosine_culled_indices.reshape(-1)
        sampled_points_world = self.transform_camera_to_world(sampled_points_cameraframe,
                                                              sampled_points_depth,
                                                              image_key)
        sampled_points_world = sampled_points_world[:3,:]

        augmented_count = 0
        depth_index = 0
        for leaf_node in self.leaf_nodes[image_key]:
            if leaf_node.unoccupied:
                if depth_rejected_indices[depth_index]:
                    leaf_node.depth_interpolated = False
                else:
                    leaf_node.sampled_point_depth = sampled_points_depth[:, depth_index]
                    leaf_node.sampled_point_world = sampled_points_world[:, depth_index]
                    leaf_node.depth_interpolated = True
                    # print(original_indices.shape, indices.shape)
                    leaf_node.sampled_point_neighbours_indices = original_indices[:, indices[depth_index]]
                    leaf_node.sampled_point_neighbours_uv = pix_coords[:, indices[depth_index]]
                    augmented_count += 1
                depth_index += 1
        return augmented_count
    
    def transform_camera_to_world(self, sampled_points_cameraframe, sampled_points_depth, image_key):
        sampled_points_cameraframe = np.concatenate((sampled_points_cameraframe[:2,:] * sampled_points_depth.reshape(1,-1),
                                                     sampled_points_depth.reshape(1,-1)), axis=0)
        R = self.rotations_image[image_key]
        t = self.translations_image[image_key]
        Rt = np.concatenate((R, t.reshape(3,1)), axis=1)
        Rt_4x4 = np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)
        sampled_points_cameraframe_h = np.vstack((sampled_points_cameraframe, 
                                                  np.ones((1, sampled_points_cameraframe.shape[1]))))
        sampled_points_worldframe = np.linalg.inv(Rt_4x4) @ sampled_points_cameraframe_h
        return sampled_points_worldframe[:3,:]

    def compute_depth(self, sampled_points_cameraframe, a, b, c, d):
        direction_vectors = sampled_points_cameraframe
        t = -d.reshape(1,-1) / np.sum(np.concatenate((a.reshape(1,-1),
                                                      b.reshape(1,-1),
                                                      c.reshape(1,-1)), axis=0) * direction_vectors, 
                                                      axis=0).reshape(1,-1)
        normal_vectors = np.concatenate((a.reshape(1,-1), b.reshape(1,-1), c.reshape(1,-1)), axis=0)
        cosine_similarity = np.abs(np.sum(normal_vectors * direction_vectors, axis=0)) /\
                                   np.linalg.norm(normal_vectors, axis=0).reshape(1,-1) /\
                                   np.linalg.norm(direction_vectors,axis=0).reshape(1,-1)
        cosine_culled_indices = cosine_similarity < self.cosine_threshold
        depth = t.reshape(1,-1)*direction_vectors[2:]
        return depth, cosine_culled_indices
        

    def compute_normal_vector(self, pix_coords_cameraframe, depths, indices):
        pix_coords_cameraframe = np.concatenate((pix_coords_cameraframe[:2,:]*depths.reshape(1,-1),
                                                 depths.reshape(1,-1)), axis=0)
        p1 = pix_coords_cameraframe[:, indices[:, 0]]
        p2 = pix_coords_cameraframe[:, indices[:, 1]]
        p3 = pix_coords_cameraframe[:, indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        # print(f"v1: {v1.shape}, v2: {v2.shape}, p1: {p1.shape}")
        normals = np.cross(v1.T, v2.T).T
        a, b, c = normals[0], normals[1], normals[2]
        d = -a * p1[0] - b * p1[1] - c * p1[2]
        return a, b, c, d

    def check_points_in_leaf_node(self, image_key, pix_coords, depths, near_culled_indices):
        rect_index = rtree.index.Index()
        rectangles = [((node.x0, node.y0, node.x0 + node.width, node.y0 + node.height), i) \
                      for i, node in enumerate(self.leaf_nodes[image_key])]
        for rect, i in rectangles:
            rect_index.insert(i, rect)
        rectangle_indices = []
        for i, (x, y) in enumerate(pix_coords.T):
            if near_culled_indices[i]:
                continue
            point = Point(x, y)
            matches = list(rect_index.intersection((x, y, x, y)))
            for match in matches:
                if box(*rectangles[match][0]).contains(point):
                    rectangle_indices.append([match, i])

        for index, points3D_idx in rectangle_indices:
            self.leaf_nodes[image_key][index].unoccupied = False
            self.leaf_nodes[image_key][index].bounded_points3d_indices.append(points3D_idx)
            self.leaf_nodes[image_key][index].bounded_points3d_depths.append(depths[0, points3D_idx])
            self.leaf_nodes[image_key][index].bounded_points3d_rgb.append(self.points3D_rgb[:, points3D_idx])
    

    def collect_leaf_nodes(self, node, image_key):
        if not node.children:
            self.leaf_nodes[image_key].append(node)
        else:
            for child in node.children:
                self.collect_leaf_nodes(child, image_key)

    def project_points3D_to_image(self, image_key):
        K = self.intrinsics_camera[self.colmap_images[image_key].camera_id]
        R = self.rotations_image[image_key]
        t = self.translations_image[image_key]
        P = K @ np.concatenate((R, t.reshape(3,1)), axis=1)
        # Convert points3D to homogeneous coordinates and project
        p3D_h = np.vstack((self.points3D, np.ones((1, self.points3D.shape[1]))))
        p3D_pix = P @ p3D_h
        p3D_depth = p3D_pix[2:3]  # Depth is the third row of the projected points 
        p3D_pix = p3D_pix[:2] / p3D_depth  # Convert to pixel space coordinates
        # Note that depth does not affected by the camera intrinsics
        return p3D_pix[:2], p3D_depth
    
    def project_world_to_image(self, image_key, coords3D, width, height):
        """
        Project 3D coordinates in world frame to image frame.
        coords3D: (3, N) array of 3D coordinates in world frame.
        """
        K = self.intrinsics_camera[self.colmap_images[image_key].camera_id]
        R = self.rotations_image[image_key]
        t = self.translations_image[image_key]
        P = K @ np.concatenate((R, t.reshape(3,1)), axis=1)
        coords3D_h = np.vstack((coords3D, np.ones((1, coords3D.shape[1]))))
        coords2D_h = P @ coords3D_h
        coords2D = coords2D_h[:2] / coords2D_h[2:3]
        depths = coords2D_h[2:3]  # Depth is the third row
        culled_indices = self.camera_frustum_culling_pix(width, height, coords2D, depths)
        return coords2D, depths, culled_indices
    
    def camera_frustum_culling_pix(self, width, height, pix_coords, depths):
        near_culled_indices = np.zeros(pix_coords.shape[1], dtype=bool)
        # print((pix_coords[0]<0).shape)
        # print((pix_coords[0]>=width).shape)
        # print((pix_coords[1]<0).shape)
        # print((pix_coords[1]>=height).shape)
        # print((depths<0).shape)
        # print((depths<0).reshape(-1).shape)
        # print(np.isnan(pix_coords).any(axis=0).reshape(-1).shape)
        near_culled_indices |= (pix_coords[0] < 0) | \
                              (pix_coords[0] >= width) | \
                              (pix_coords[1] < 0) | \
                              (pix_coords[1] >= height) | \
                              (depths < 0).reshape(-1) | \
                              np.isnan(pix_coords).any(axis=0)
        
        return near_culled_indices.reshape(-1)
    
    def visibility_culling(self, image, pix_coords, near_culled_indices):
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
        # Check if the RGB values are already normalized within [0, 1]
        if np.max(image_rgb) <= 1.0 and np.min(image_rgb) >= 0.0:
           pass
        else:
            image_rgb = image_rgb / 255.0
        if np.max(colmap_rgb) <= 1.0 and np.min(colmap_rgb) >= 0.0:
            pass
        else:
            colmap_rgb = colmap_rgb / 255.0
        diff = np.linalg.norm(image_rgb.reshape(-1) - colmap_rgb.reshape(-1))
        return diff

    def find_corresponding_leaf_node(self, image_key, pix_coord, scale=1.0):
        pix_coord = np.array(pix_coord) / scale
        root = self.roots[image_key]
        return self._find_leaf_recursive(root, pix_coord)

    def _find_leaf_recursive(self, node, pix_coord):
        # 자식이 없으면(leaf) 이 노드 반환
        if not node.children:
            return node
        # 자식 노드 중 포함되는 곳으로 재귀
        for child in node.children:
            if (child.x0 <= pix_coord[0] < child.x0 + child.width and
                child.y0 <= pix_coord[1] < child.y0 + child.height):
                return self._find_leaf_recursive(child, pix_coord)
        return None

    def construct_closest_n_views(self, n=12):
        self.closest_n_views = {}
        self.name_to_key = {self.colmap_images[key].name: key for key in self.image_keys}
        for view in self.image_keys:
            view_name = self.colmap_images[view].name
            # print(view_name)
            # print(self.image_clustering.W_dict.keys())
            similarities = [(name, self.image_clustering.W_dict[view_name][name]) for name in self.image_clustering.W_dict[view_name].keys()]
            similarities = [sim for sim in similarities if sim[0] != view_name]
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            closest_n_views = [self.name_to_key[name] for name, _ in similarities[:n]]
            self.closest_n_views[view] = closest_n_views
    
    def pixel_length_to_world_length(self, depth, pixel_length, image_key):
        """
        Convert pixel length to world length using the depth of the point and the camera intrinsics.
        depth: Depth of the point in the camera frame.
        pixel_length: Length in pixels to be converted.
        image_key: Key of the image to get the camera intrinsics.
        """
        K = self.intrinsics_camera[self.colmap_images[image_key].camera_id]
        focal_length = K[0, 0]
        # Convert pixel length to world length using the formula:
        # world_length = (depth * pixel_length) / focal_length
        world_length = (depth * pixel_length) / focal_length
        return world_length

    def correspondence_check(self):
        """
        Check the correspondence between 1) sampled points and 3D points,
        and 2) sampled points and other sampled points.
        The correspondence between two points is defined by the depth error and RGB difference.

        For case 1):
        1. Find the 3D points that are bounded within a 1-pixel circle around the sampled point.
        2. Compute the depth error and RGB difference.
        3. The depth threshold is set to the 1-pixel radius in world coordinates.
        1-a) If the depth error is less than the threshold,
             the sampled point is representing the geometry redundantly, so it is unnecessary.
        1-b) If the depth error is larger than the threshold,
             the sampled point represents the geometry at a deeper position, so abort pruning.

        For case 2):
        1. Project the sampled points from one view to the other views.
        2. Find the leaf node of the other view that contains the projected point.
        3. Compute the depth error and RGB difference between the sampled point
           of the leaf node in the other view and the projected point.
        2-a) If the error is less than the threshold,
             both sampled points are representing the same geometry,
             indicating that they are corresponding points.
             Further steps will be taken to merge the corresponding sampled points.
        2-b) If the error is larger than the threshold,
             each sampled point represents the geometry at a different position,
             so they are not corresponding points at that moment.
             Therefore, we will keep both sampled points (do nothing).

        There is a special case where the corresponding leaf node in the other view
        does not contain an SfM 3D point nor a sampled point.
        In that case, we will do nothing.
        """
        corresponding_viewkey_nodeidx_pairs = []
        # [[target_view, target_node_index, ref_view, ref_node_index], ...]
        for view in tqdm(self.image_keys, desc="Correspondence Check"):
            sampled_points_world = []
            sampled_points_rgb = []
            sampled_points_target_node_indices = []

            for i, leaf_node in enumerate(self.leaf_nodes[view]):
                if leaf_node.sampled_point_depth is None or not leaf_node.depth_interpolated:
                    continue
                sampled_point_world = leaf_node.sampled_point_world
                if sampled_point_world is None or np.isnan(sampled_point_world).any():
                    continue
                sampled_point_rgb = leaf_node.sampled_point_rgb
                sampled_points_world.append(sampled_point_world)
                sampled_points_rgb.append(sampled_point_rgb)
                sampled_points_target_node_indices.append(i)
            sampled_points_world = np.array(sampled_points_world).T
            sampled_points_rgb = np.array(sampled_points_rgb)
            sampled_points_target_node_indices = np.array(sampled_points_target_node_indices)
            for ref_view in self.closest_n_views[view]:
                sampled_points_coords2d_ref, sampled_points_depth_ref, culled_indices = \
                    self.project_world_to_image(ref_view, sampled_points_world, self.roots[ref_view].width, self.roots[ref_view].height)
                sampled_points_coords2d_ref = sampled_points_coords2d_ref[:, ~culled_indices]
                sampled_points_depth_ref = sampled_points_depth_ref[:, ~culled_indices]
                sampled_points_rgb_ref = sampled_points_rgb[~culled_indices]
                sampled_points_target_node_indices_ref = sampled_points_target_node_indices[~culled_indices]
                for i, sampled_point in enumerate(sampled_points_coords2d_ref.T):
                    ref_leaf_node = self.find_corresponding_leaf_node(ref_view, sampled_point)
                    assert ref_leaf_node is not None
                    # Check if there is a 3D point in the leaf node
                    if ref_leaf_node.bounded_points3d_indices:
                        # # Check if there is a 3D point that is close enough to the sampled point
                        # distances = np.linalg.norm(ref_leaf_node.bounded_points3d_coords2d - sampled_point.reshape(2, 1), axis=0)
                        # close_indices = np.where(distances < 1.0)[0]  # 1 pixel radius in image space
                        # if len(close_indices) > 0:j
                        
                        #     # Correspondence found with a 3D point
                        #     leaf_node.matching_inference_count += 1
                        #     for idx in close_indices:
                        #         depth_error = np.abs(ref_leaf_node.bounded_points3d_depths[idx] - sampled_points_depth_ref[i])
                        #         rgb_diff = self.pixelwise_rgb_diff(ref_leaf_node.bounded_points3d_rgb[:, idx], sampled_points_rgb_ref[i])
                        #         if depth_error < self.depth_cutoff and rgb_diff < self.rgb_diff_threshold:
                        #             # Correspondence found with a 3D point, log that it should be pruned
                        #             leaf_node.matching_log[ref_view] = "a1"
                        #         else:
                        #             leaf_node.matching_log[ref_view] = "a2"
                        pass
                    elif ref_leaf_node.sampled_point_depth is not None:
                        # Check if the sampled point and its corresponding sampled point in the other view
                        # are close enough in depth and RGB
                        depth_error = np.abs(ref_leaf_node.sampled_point_depth - sampled_points_depth_ref[0,i])
                        rgb_diff = self.pixelwise_rgb_diff(ref_leaf_node.sampled_point_rgb, sampled_points_rgb_ref[i])
                        depth_bound = ref_leaf_node.sampled_point_depth * np.tan(np.deg2rad(0.5))
                        # depth_bound = self.pixel_length_to_world_length(ref_leaf_node.sampled_point_depth, max(ref_leaf_node.width, ref_leaf_node.height), ref_view)

                        if depth_error < depth_bound and rgb_diff < self.rgb_diff_threshold:
                            leaf_node.matching_log[ref_view] = "b1"
                            corresponding_viewkey_nodeidx_pairs.append([view, sampled_points_target_node_indices_ref[i], ref_view, self.leaf_nodes[ref_view].index(ref_leaf_node)])
                        else:
                            leaf_node.matching_log[ref_view] = "b2"
        # now we have the correspondence pairs
        # We now gather the sampled points in case of b1
        # First gather the points in B1
        # Then assign covariances of corresponding leaf nodes w.r.t sample point
        current_view = None
        current_node = None
        sample_point_count = -1
        final_sample_points_world = []
        final_sample_points_rgb = []
        final_sample_points_indices = []
        final_image_ids = []
        final_cov_labels = []
        import pdb
        # pdb.set_trace()
        corresponding_viewkey_nodeidx_pairs = sorted(corresponding_viewkey_nodeidx_pairs, key=lambda x: (x[0], x[1]))
        # 양방향 엣지 찾기
        bidirectional_edges = []
        edge_set = set()

        # 먼저 모든 엣지를 edge_set에 추가
        for edge in corresponding_viewkey_nodeidx_pairs:
            v1, n1, v2, n2 = edge
            edge_set.add((v1, n1, v2, n2))

        # 양방향 엣지 찾기
        processed = set()
        for edge in corresponding_viewkey_nodeidx_pairs:
            v1, n1, v2, n2 = edge
            edge_tuple = (v1, n1, v2, n2)
            reverse_edge = (v2, n2, v1, n1)
            
            # 역방향 엣지가 존재하고 아직 처리되지 않았다면
            if reverse_edge in edge_set and edge_tuple not in processed and reverse_edge not in processed:
                bidirectional_edges.append([v1, n1, v2, n2])
                processed.add(edge_tuple)
                processed.add(reverse_edge)

        # 결과 저장
        self.bidirectional_edges = bidirectional_edges
        def find_maximal_cliques_from_bidirectional_edges(bidirectional_edges):
            """
            양방향 엣지들로부터 maximal clique들을 찾는 함수
            
            Args:
                bidirectional_edges: [[v1, n1, v2, n2], ...] 형태의 양방향 엣지 리스트
            
            Returns:
                maximal_cliques: 각 clique는 [(view, node), ...] 형태의 노드 리스트
            """
            from collections import defaultdict
            
            # 그래프 구축: 각 노드의 이웃 노드들을 저장
            graph = defaultdict(set)
            all_nodes = set()
            
            # 양방향 엣지로부터 그래프 구축
            for v1, n1, v2, n2 in bidirectional_edges:
                node1 = (v1, n1)
                node2 = (v2, n2)
                graph[node1].add(node2)
                graph[node2].add(node1)
                all_nodes.add(node1)
                all_nodes.add(node2)
            
            def is_clique(nodes):
                """주어진 노드들이 clique를 형성하는지 확인"""
                nodes_list = list(nodes)
                for i in range(len(nodes_list)):
                    for j in range(i + 1, len(nodes_list)):
                        if nodes_list[j] not in graph[nodes_list[i]]:
                            return False
                return True
            
            def bron_kerbosch(R, P, X, cliques):
                """Bron-Kerbosch 알고리즘으로 maximal clique 찾기"""
                if not P and not X:
                    if len(R) >= 2:  # 최소 2개 노드로 구성된 clique만 저장
                        cliques.append(list(R))
                    return
                
                # Pivot 선택 (P ∪ X에서 가장 많은 이웃을 가진 노드)
                pivot = None
                max_connections = -1
                for node in P.union(X):
                    connections = len(graph[node].intersection(P))
                    if connections > max_connections:
                        max_connections = connections
                        pivot = node
                
                # P에서 pivot의 이웃이 아닌 노드들에 대해 재귀 호출
                for node in list(P - graph[pivot]):
                    neighbors = graph[node]
                    bron_kerbosch(
                        R.union({node}),
                        P.intersection(neighbors),
                        X.intersection(neighbors),
                        cliques
                    )
                    P.remove(node)
                    X.add(node)
            
            # Bron-Kerbosch 알고리즘 실행
            maximal_cliques = []
            bron_kerbosch(set(), all_nodes.copy(), set(), maximal_cliques)
            
            return maximal_cliques

        maximal_cliques = find_maximal_cliques_from_bidirectional_edges(bidirectional_edges)
        # import pdb; pdb.set_trace()
        
        self.merged_correspondences = maximal_cliques
        self.merged_sample_points_world = []
        self.merged_sample_points_rgb = []
        self.merged_sample_points_indices = []
        self.merged_image_ids = []
        self.mean_labels = []
        self.cov_labels = []
        self.rgb_labels = []

        # 4. Merge sampled points in the leaf nodes of the corresponding views
        for i, group in enumerate(self.merged_correspondences):
            sampled_points_world_raw = [self.leaf_nodes[v][n].sampled_point_world.T for v, n in group]
            sampled_points_rgb_raw = [self.leaf_nodes[v][n].sampled_point_rgb for v, n in group]

            # Compute the mean of the sampled points
            self.merged_sample_points_world.append(np.mean(sampled_points_world_raw, axis=0))
            self.merged_sample_points_rgb.append(np.mean(sampled_points_rgb_raw, axis=0))

            for v, n in group:
                mean = self.leaf_nodes[v][n].sampled_point_uv
                # cov = [[self.leaf_nodes[v][n].width**2/36, 0], [0, self.leaf_nodes[v][n].height**2/36]]
                ix = self.ix[v][mean[1], mean[0]]
                iy = self.iy[v][mean[1], mean[0]]

                if ix < 1e-3 and iy < 1e-3:
                    cov = np.array([[self.leaf_nodes[v][n].width**2 / 36, 0], [0, self.leaf_nodes[v][n].height**2 / 36]])
                elif ix < 1e-3 and iy >= 1e-3:
                    cov = np.array([[self.leaf_nodes[v][n].width**2 / 36 / 9, 0], [0, self.leaf_nodes[v][n].height**2 / 36]])
                elif ix >= 1e-3 and iy < 1e-3:
                    cov = np.array([[self.leaf_nodes[v][n].width**2 / 36, 0], [0, self.leaf_nodes[v][n].height**2 / 36 / 9]])
                else:
                    structure_tensor = np.array([[ix*ix, ix*iy], [ix*iy, iy*iy]])
                    eigvals, eigvecs = np.linalg.eigh(structure_tensor)
                    eigvals_max = np.max(eigvals)
                    eigvals = np.maximum(eigvals/(eigvals_max), 0.33)
                    eigvals = np.sqrt(eigvals) * self.leaf_nodes[v][n].width / 6
                    scale_matrix = np.diag([eigvals[1], eigvals[0]])
                    cov = eigvecs @ scale_matrix @ scale_matrix @ eigvecs.T

                self.merged_image_ids.append(v)
                self.merged_sample_points_indices.append(i)
                self.mean_labels.append(mean)
                self.cov_labels.append(cov)
                self.rgb_labels.append(self.leaf_nodes[v][n].sampled_point_rgb)

        # pdb.set_trace()
        
        print(f"Total number of merged sampled points: {len(self.merged_sample_points_world)}")
        print(f"Total number of cliques: {len(self.merged_correspondences)}")

    def build_conflict_free_groups(self, component):
        """Build maximal conflict-free groups from a component"""
        from collections import defaultdict
        
        # Group by view
        view_to_nodes = defaultdict(list)
        for v, n in component:
            view_to_nodes[v].append((v, n))
        
        if all(len(nodes) == 1 for nodes in view_to_nodes.values()):
            # No conflicts - return as single group
            return [component]
        
        # Build all possible conflict-free combinations
        views = list(view_to_nodes.keys())
        all_combinations = []
        
        def generate_combinations(view_idx, current_combo):
            if view_idx == len(views):
                if len(current_combo) >= 2:  # 최소 2개 노드 필요
                    all_combinations.append(current_combo.copy())
                return
            
            view = views[view_idx]
            nodes = view_to_nodes[view]
            
            # Option 1: Skip this view
            generate_combinations(view_idx + 1, current_combo)
            
            # Option 2: Select one node from this view
            for node in nodes:
                current_combo.append(node)
                generate_combinations(view_idx + 1, current_combo)
                current_combo.pop()
        
        generate_combinations(0, [])
        
        # Find maximal non-overlapping combinations
        all_combinations.sort(key=len, reverse=True)
        used_nodes = set()
        result_groups = []
        
        for combination in all_combinations:
            # Check if any node in this combination is already used
            if any(node in used_nodes for node in combination):
                continue
            
            # Add this combination as a maximal group
            result_groups.append(combination)
            used_nodes.update(combination)
        
        return result_groups