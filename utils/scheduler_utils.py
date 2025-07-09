import numpy as np
from random import randint
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import sys, os
sys.path.append("/mnt/disk2/auggs/gaussian-splatting")
from utils.bundle_utils import build_covisibility_matrix
from utils.colmap_utils import get_colmap_data
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

class GroupScheduler:
    """
    Gaussian Splatting Scheduler
    Given a set of grouped indices, the scheduler will return a uid for each iteration.
    Also, the scheduler will trigger a densify_and_prune or reset_opacity at the appropriate steps.
    
    The training schedule in the training of 3D Gaussian splatting is composed of three stages:
    (1) Warmup stage: From the start until 500 iterations, randomly access to cameras.
    (2) Densification stage: From the end of warmup stage until mid-training,
        multiple ADC intervals are applied.
    (3) Finetuning stage: From the end of densification stage until the maximum iterations,
        randomly access to cameras.
    Each ADC interval is consisted of an ADC batch,
    and there are three types of ADC batches: sequential, grouped random, and random.
    (a) Sequential: A single batch of camera sequence.
        The size of the batch is the total number of cameras.
    (b) Grouped Random: A single batch is consisted of single group,
        and each batch is consisted of random cameras from the group.
        The size of the batch is determined by the number of cameras in the group.
    (c) Random: A single batch is consisted of random cameras.
        The size of the batch is hard-coded as 100.

    The scheduler takes ordered groups as a dictionary,
    the keys are the group indices and the values are the ordered camera names within the group.
    """
    def __init__(self, 
                 cameras,
                 grouped_names: dict, 
                 densify_until_iter: int, 
                 densify_from_iter: int, 
                 debug: bool=False,
                 ):
        self.debug = debug
        if self.debug and not grouped_names:
            self.grouped_names = {0: [i for i in range(42)], 1: [i for i in range(42, 86)], 2: [i for i in range(86, 120)],
                                  3: [i for i in range(120, 150)], 4: [i for i in range(150, 200)]}
        else:
            self.grouped_names = grouped_names
        self.n_groups = len(self.grouped_names)
        self.densify_until_iter = densify_until_iter
        self.densify_from_iter = densify_from_iter
        self.generate_dict_name_to_uid(cameras)
        self.generate_ordered_uids()
        self.uid_stack = None
        self.sequential_count = 0
        self.group_idx = 0
        self.densify_and_prune_flag = False
        self.reset_opacity_flag = False
        self.random_group_uid_stack = None
        self.num_turns = 20

    def set_num_turns(self, num_turns: int):
        self.num_turns = num_turns

    def generate_dict_name_to_uid(self, cameras):
        """
        Generate a dictionary that maps the image name to the uid.
        """
        if self.debug:
            self.name_to_uid = {}
            for group_name_list in self.grouped_names.values():
                for name in group_name_list:
                    self.name_to_uid[name] = len(self.name_to_uid)
        else:   
            self.name_to_uid = {cam.image_name: cam.uid for cam in cameras}

    def generate_ordered_uids(self):
        """
        Generate a list of uids that are ordered by the group indices.
        """
        ordered_uids = []
        for group_index in self.grouped_names.keys():
            ordered_uids.extend(self.name_to_uid[name] for name in self.grouped_names[group_index])
        print(ordered_uids)
        self.ordered_uids = ordered_uids

    def scheduled_training_index(self, iteration: int,):
        """
        For given iteration, return the uid of the camera to be accessed.
        Also, the scheduler will trigger a densify_and_prune or reset_opacity at the appropriate steps.
        The iteration range starts from 1 to the maximum iterations(30000).
        """
        if iteration <= self.densify_from_iter or iteration > self.densify_until_iter:
            # Warmup stage
            if not self.uid_stack:
                self.uid_stack = list(self.name_to_uid.values())
            rand_idx = randint(0, len(self.uid_stack) - 1)
            vind = self.uid_stack.pop(rand_idx)
            # print(f"Iteration: {iteration}, Warmup stage: {vind}")
            return vind
        elif iteration > self.densify_from_iter and iteration <= self.densify_until_iter:
            if iteration <= self.densify_from_iter + len(self.ordered_uids)*self.num_turns:
                if (iteration - self.densify_from_iter) % len(self.ordered_uids) == 1:
                    interval = ((iteration - self.densify_from_iter)//len(self.ordered_uids))*(len(self.ordered_uids)//5)
                    first_part = self.ordered_uids.copy()[interval:]
                    second_part = self.ordered_uids.copy()[:interval]
                    self.uid_sequence = first_part + second_part
                    if self.sequential_count % 2 != 0:
                        self.uid_sequence = self.uid_sequence[::-1]
                    self.sequential_count += 1
                if (iteration - self.densify_from_iter) % len(self.ordered_uids) == 0:
                    self.densify_and_prune_flag = True
                if iteration == self.densify_from_iter + len(self.ordered_uids)*self.num_turns:
                    self.reset_opacity_flag = True
                vind = self.uid_sequence[(iteration - self.densify_from_iter-1) % len(self.uid_sequence)]
                # print(f"Iteration: {iteration}, Sequential stage: {vind}, densification_flag = {self.densify_and_prune_flag}, reset_opacity_flag = {self.reset_opacity_flag}")
                return vind
            elif iteration > self.densify_from_iter + len(self.ordered_uids)*self.num_turns \
                and iteration <= self.densify_from_iter + len(self.ordered_uids)*2*self.num_turns:
                if not self.random_group_uid_stack:
                    self.group_names = self.grouped_names[self.group_idx]
                    self.group_uids = [self.name_to_uid[name] for name in self.group_names]
                    self.random_group_uid_stack = self.group_uids.copy()
                if len(self.random_group_uid_stack) == 1:
                    self.group_idx = randint(0, self.n_groups - 1)
                if (iteration - self.densify_from_iter - len(self.ordered_uids)*self.num_turns) % len(self.ordered_uids) == 0:
                    self.densify_and_prune_flag = True
                if iteration == self.densify_from_iter + len(self.ordered_uids)*2*self.num_turns:
                    self.reset_opacity_flag = True
                rand_idx = randint(0, len(self.random_group_uid_stack) - 1)
                vind = self.random_group_uid_stack.pop(rand_idx)
                # print(f"Iteration: {iteration}, Grouped Random stage: {vind}, densification_flag = {self.densify_and_prune_flag}, reset_opacity_flag = {self.reset_opacity_flag}")
                return vind
            else:
                if not self.uid_stack:
                    self.uid_stack = list(self.name_to_uid.values())
                if (iteration - self.densify_from_iter - len(self.ordered_uids)*2*self.num_turns) % 100 == 0:
                    self.densify_and_prune_flag = True
                if (iteration - self.densify_from_iter - len(self.ordered_uids)*2*self.num_turns) % 3000 == 0:
                    self.reset_opacity_flag = True
                rand_idx = randint(0, len(self.uid_stack) - 1)
                vind = self.uid_stack.pop(rand_idx)
                # print(f"Iteration: {iteration}, Random densification stage: {vind}, densification_flag = {self.densify_and_prune_flag}, reset_opacity_flag = {self.reset_opacity_flag}")
                return vind
        else:
            if not self.uid_stack:
                self.uid_stack = list(self.name_to_uid.values())
            rand_idx = randint(0, len(self.uid_stack) - 1)
            vind = self.uid_stack.pop(rand_idx)
            # print(f"Iteration: {iteration}, Random stage: {vind}, densification_flag = {self.densify_and_prune_flag}, reset_opacity_flag = {self.reset_opacity_flag}")
            return vind
        
class PartialGroupScheduler:
    def __init__(self, 
                 cameras,
                 grouped_names: dict, 
                 densify_until_iter: int,
                 densify_from_iter: int,
                 mi_warmup: bool=False,
                 bb: bool=False,
                 similarity_grouping: bool=False,
                 clustering=None,
                 debug: bool=False,
                 ):
        self.debug = debug
        if self.debug and not grouped_names:
            self.grouped_names = {0: [i for i in range(42)], 1: [i for i in range(42, 86)], 2: [i for i in range(86, 120)],
                                  3: [i for i in range(120, 150)], 4: [i for i in range(150, 200)]}
        else:
            self.grouped_names = grouped_names
        self.n_groups = len(self.grouped_names)
        self.densify_until_iter = densify_until_iter
        self.densify_from_iter = densify_from_iter
        self.generate_dict_name_to_uid(cameras)
        self.generate_ordered_uids()
        self.uid_stack = None
        self.sequential_count = 0
        self.group_idx = 0
        self.densify_and_prune_flag = False
        self.reset_opacity_flag = False
        self.random_group_uid_stack = None
        self.mi_warmup = mi_warmup
        self.bb = bb
        self.similarity_grouping = similarity_grouping
        if clustering:
            self.clustering = clustering
        if similarity_grouping:
            self.assign_closest_n_views(n=20)

    def generate_dict_name_to_uid(self, cameras):
        """
        Generate a dictionary that maps the image name to the uid.
        """
        if self.debug:
            self.name_to_uid = {}
            for group_name_list in self.grouped_names.values():
                for name in group_name_list:
                    self.name_to_uid[name] = len(self.name_to_uid)
        else:   
            self.name_to_uid = {cam.image_name: cam.uid for cam in cameras}
        self.uid_to_name = {v: k for k, v in self.name_to_uid.items()}

    def generate_ordered_uids(self):
        """
        Generate a list of uids that are ordered by the group indices.
        """
        ordered_uids = []
        for group_index in self.grouped_names.keys():
            ordered_uids.extend(self.name_to_uid[name] for name in self.grouped_names[group_index])
        print(ordered_uids)
        self.ordered_uids = ordered_uids

    def scheduled_training_index(self, iteration: int,):
        """
        For given iteration, return the uid of the camera to be accessed.
        Also, the scheduler will trigger a densify_and_prune or reset_opacity at the appropriate steps.
        The iteration range starts from 1 to the maximum iterations(30000).
        """
        if iteration % 3000 == 0:
            self.reset_opacity_flag = True
        if iteration <= self.densify_from_iter or iteration > self.densify_until_iter:
            if self.mi_warmup and iteration <= self.densify_from_iter:
                if not self.uid_stack:
                    self.uid_stack = list(self.name_to_uid.values())
                if iteration % 2 == 1:
                    rand_idx = randint(0, len(self.uid_stack) - 1)
                    self.vind = self.uid_stack.pop(rand_idx)
                    return self.vind
                else:
                    if self.bb:
                        # 현재 뷰의 이름
                        curr_view_name = self.uid_to_name[self.vind]
                        
                        # 남은 뷰들의 이름
                        available_names = [self.uid_to_name[uid] for uid in self.uid_stack]
                        
                        # 현재 뷰와 남은 뷰들 간의 유사도 계산
                        similarities = [(name, self.clustering.W_dict[curr_view_name][name]) for name in available_names]
                        min_sim_name, min_sim_val = min(similarities, key=lambda x: x[1])
                        # print(f"bb=True - Current view: {curr_view_name}, Least similar view: {min_sim_name}, whatifbb=False: {self.clustering.find_least_similar_view(self.uid_to_name[self.vind])} Similarity: {min_sim_val}")
                        
                        mi_name = min_sim_name
                        mi_ind = self.name_to_uid[mi_name]
                        self.uid_stack.remove(mi_ind)
                        return mi_ind
                    else:
                        mi_name = self.clustering.find_least_similar_view(self.uid_to_name[self.vind])
                        # print(f"bb=False - Current view: {self.uid_to_name[self.vind]}, Least similar view: {mi_name}, Similarity: {self.clustering.W_dict[self.uid_to_name[self.vind]][mi_name]}")
                        mi_ind = self.name_to_uid[mi_name]
                        return mi_ind
            else:
                if not self.uid_stack:
                    self.uid_stack = list(self.name_to_uid.values())
                rand_idx = randint(0, len(self.uid_stack) - 1)
                vind = self.uid_stack.pop(rand_idx)
                return vind
        else:
            # Densification stage
            iteration_in_stage = iteration - self.densify_from_iter
            stage = iteration_in_stage // 100
            step_in_stage = iteration_in_stage % 100

            # 매 100번째 iteration마다 densification 수행
            if step_in_stage == 0:
                self.densify_and_prune_flag = True
            
            # 매 100번째 iteration의 다음 iteration에서 그룹 변경
            if step_in_stage == 1:
                if self.similarity_grouping:
                    self.group_seed = randint(0,len(self.name_to_uid.keys())-1)
                    self.group_seed_name = self.uid_to_name[self.group_seed]
                    self.group_names = self.closest_n_views[self.group_seed_name]
                    self.group_uids = [self.name_to_uid[name] for name in self.group_names]
                    self.random_group_uid_stack = None
                else:
                    self.group_idx = stage % self.n_groups
                    self.group_names = self.grouped_names[self.group_idx]
                    self.group_uids = [self.name_to_uid[name] for name in self.group_names]
                    self.random_group_uid_stack = None

            # 앞 80회는 warmup과 동일
            if step_in_stage < 80:
                if not self.uid_stack:
                    self.uid_stack = list(self.name_to_uid.values())
                rand_idx = randint(0, len(self.uid_stack) - 1)
                vind = self.uid_stack.pop(rand_idx)
                return vind
            # 뒤 20회는 현재 그룹 내에서 비복원추출
            else:
                if not self.random_group_uid_stack or len(self.random_group_uid_stack) == 0:
                    self.random_group_uid_stack = self.group_uids.copy()
                rand_idx = randint(0, len(self.random_group_uid_stack) - 1)
                vind = self.random_group_uid_stack.pop(rand_idx)
                return vind
            
    def assign_ImageClustering(self, clustering):
        self.clustering = clustering

    def find_least_similar_view(self, view_name: str) -> str:
        """
        주어진 시점과 가장 affinity가 낮은 시점을 찾습니다.
        """
        # print(f"\nfind_least_similar_view for {view_name}")
        similarities = [(name, self.W_dict[view_name][name]) 
                       for name in self.W_dict[view_name].keys() 
                       if name != view_name]
        # print("All similarities:", sorted(similarities, key=lambda x: x[1])[:5])
        result = min(similarities, key=lambda x: x[1])[0]
        # print(f"Selected: {result} with similarity {self.W_dict[view_name][result]}")
        return result
    
    def assign_closest_n_views(self, n=20):
        self.closest_n_views = {}
        for view_name in self.clustering.W_dict.keys():
            similarities = [(name, self.clustering.W_dict[view_name][name]) for name in self.clustering.W_dict[view_name].keys()]
            similarities = [sim for sim in similarities if sim[0] != view_name]
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            closest_n_views = [name for name, _ in similarities[:n]]
            self.closest_n_views[view_name] = closest_n_views

class ImageClustering:
    def __init__(self,
                 dataset_path,
		 n_clusters = None,
         inv_affinity_matrix = False):
        self.dataset_path = dataset_path
        self.n_clusters = n_clusters
        self.inv_affinity_matrix = inv_affinity_matrix
        self.images, self.points3D, self.cameras = get_colmap_data(self.dataset_path)
        self.split_train_test()
        self.create_affinity_matrix()
        self.cluster_images()
        self.select_n_clusters()
        self.intra_cluster_ordering()
        self.cluster_ordering()

    def split_train_test(self):
        image_id_name = [[self.images[key].id, self.images[key].name] for key in self.images.keys()]
        image_id_name_sorted = sorted(image_id_name, key=lambda x: x[1])
        self.train_ids = []
        self.test_ids = []
        for i, (id, name) in enumerate(image_id_name_sorted):
            if i % 8 == 0:
                self.test_ids.append(id)
            else:
                self.train_ids.append(id)
        self.train_images = {k: v for k, v in self.images.items() if v.id in self.train_ids}
    
    def create_affinity_matrix(self):
        from utils.bundle_utils import build_covisibility_matrix  # 함수 내부에서 import
        self.affinity_matrix, self.id_to_idx, self.idx_to_id = build_covisibility_matrix(self.train_images, self.points3D)
    
    def cluster_images(self):
        W = np.array(self.affinity_matrix)
        # print("Original W shape:", W.shape)
        # print("W min, max before inv:", W.min(), W.max())
        
        if self.inv_affinity_matrix:
            W = 1 / (W+1)
        # print("W min, max after inv:", W.min(), W.max())
        
        self.W = (W - W.min()) / (W.max() - W.min())
        # print("W min, max after norm:", self.W.min(), self.W.max())
        
        # 이름 기반 affinity dictionary 생성
        self.W_dict = {}
        for id1 in self.train_images.keys():
            img_name1 = self.train_images[id1].name
            self.W_dict[img_name1] = {}
            idx1 = self.id_to_idx[id1]
            for id2 in self.train_images.keys():
                img_name2 = self.train_images[id2].name
                idx2 = self.id_to_idx[id2]
                self.W_dict[img_name1][img_name2] = self.W[idx1, idx2]
            
        # W_dict 값 확인
        # print("\nSample W_dict values:")
        sample_img = next(iter(self.W_dict))
        # print(f"Values for {sample_img}:")
        # print(sorted([(k,v) for k,v in self.W_dict[sample_img].items()], key=lambda x: x[1])[:5])
        
        # train_images의 순서를 리스트로 유지
        train_image_list = list(self.train_images.items())
        
        D = np.diag(self.W.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(self.W.sum(axis=1), 1e-10)))
        L_sym = np.eye(len(self.W)) - D_inv_sqrt @ self.W @ D_inv_sqrt
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(L_sym)

    def select_n_clusters(self):
        if not self.n_clusters:
            self.score = []
            for i in range(2, 50):
                n_clusters = i
                U = self.eigenvectors[:, :n_clusters]
                U_norm = normalize(U, norm='l2')
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(U_norm)
                sscore = silhouette_score(U_norm, clusters)
                self.score.append(sscore)
                # print(f"{i} clusters, silhouette score: {sscore}")
            self.n_clusters = int(input("select number of clusters: "))
        U = self.eigenvectors[:, :self.n_clusters]
        U_norm = normalize(U, norm='l2')
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(U_norm)
    
    def intra_cluster_ordering(self):
        self.cluster_ids = np.unique(self.clusters)
        self.intra_cluster_ordering = {}

        for cid in self.cluster_ids:
            indices = np.where(self.clusters==cid)[0]
            sub_adj = self.W[np.ix_(indices, indices)]
            degrees = sub_adj.sum(axis=1)
            start_node_idx = np.argmax(degrees)

            ordered_indices = [start_node_idx]
            remaining_indices = set(range(len(indices)))
            remaining_indices.remove(start_node_idx)

            current_idx = start_node_idx
            while remaining_indices:
                adjacencies = sub_adj[current_idx, list(remaining_indices)]

                if adjacencies.max() == 0:
                    next_idx = max(remaining_indices, key=lambda x: degrees[x])
                else:
                    next_idx = list(remaining_indices)[np.argmax(adjacencies)]

                ordered_indices.append(next_idx)
                remaining_indices.remove(next_idx)
                current_idx = next_idx
            sorted_node_indices = indices[ordered_indices]
            self.intra_cluster_ordering[cid] = sorted_node_indices.tolist()
        for cid, ordering in self.intra_cluster_ordering.items():
            print(f"Cluster {cid} #views: {len(ordering)}")
    
    def cluster_ordering(self):
        cluster_adj_matrix = np.zeros((self.n_clusters, self.n_clusters))
        for i, ci in enumerate(self.cluster_ids):
            indices_i = np.where(self.clusters == ci)[0]
            for j, cj in enumerate(self.cluster_ids):
                if i >= j:
                    continue
                indices_j = np.where(self.clusters == cj)[0]
                inter_adj = self.W[np.ix_(indices_i, indices_j)]
                cluster_adj_matrix[i, j] = cluster_adj_matrix[j, i] = inter_adj.sum()

        self.ordered_cluster_ids = []
        remaining_cluster_ids = set(range(self.n_clusters))
        cluster_degrees = cluster_adj_matrix.sum(axis=1)
        current_cluster = np.argmax(cluster_degrees)
        self.ordered_cluster_ids.append(current_cluster)
        remaining_cluster_ids.remove(current_cluster)

        while remaining_cluster_ids:
            adjacencies = cluster_adj_matrix[current_cluster, list(remaining_cluster_ids)]
            if adjacencies.max() == 0:
                next_cluster = max(remaining_cluster_ids, key=lambda x: cluster_degrees[x])
            else:
                next_cluster = list(remaining_cluster_ids)[np.argmax(adjacencies)]

            self.ordered_cluster_ids.append(next_cluster)
            remaining_cluster_ids.remove(next_cluster)
            current_cluster = next_cluster

        self.ordered_cluster_ids = [self.cluster_ids[i] for i in self.ordered_cluster_ids]
        # print("Ordered cluster ids: ", self.ordered_cluster_ids)
        self.ordered_clusters = [self.intra_cluster_ordering[i] for i in self.ordered_cluster_ids]
        # print("Ordered clusters: ", self.ordered_clusters)
        self.ordered_colmap_ids = {}
        for i, cluster in enumerate(self.ordered_clusters):
            self.ordered_colmap_ids[i] = [self.idx_to_id[idx] for idx in cluster]
        # print("Ordered colmap ids: ", self.ordered_colmap_ids)        
        self.ordered_cluster_names = {}
        for i in range(self.n_clusters):
            self.ordered_cluster_names[i] = [self.images[id].name for id in self.ordered_colmap_ids[i]]
        # print("Ordered cluster names: ", self.ordered_cluster_names)

    def find_least_similar_view(self, view_name: str) -> str:
        """
        주어진 시점과 가장 affinity가 낮은 시점을 찾습니다.
        """
        # print(f"\nfind_least_similar_view for {view_name}")
        similarities = [(name, self.W_dict[view_name][name]) 
                       for name in self.W_dict[view_name].keys() 
                       if name != view_name]
        # print("All similarities:", sorted(similarities, key=lambda x: x[1])[:5])
        result = min(similarities, key=lambda x: x[1])[0]
        # print(f"Selected: {result} with similarity {self.W_dict[view_name][result]}")
        return result
