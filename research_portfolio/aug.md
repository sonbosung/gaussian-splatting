# Research Portfolio: Multi-View Consistent Point Cloud Augmentation

## Research Overview

**Title:** Adaptive Point Cloud Densification via Quadtree Sampling and Multi-View Geometric Consistency

**Domain:** 3D Computer Vision, Structure-from-Motion, Multi-View Geometry

**Problem Statement:** Structure-from-Motion (SfM) pipelines like COLMAP produce sparse 3D reconstructions with significant gaps in regions where feature matching fails—particularly in textureless areas, repetitive patterns, or low-overlap zones. These sparse outputs limit downstream applications requiring dense geometric representations.

**Research Question:** How can we augment sparse SfM point clouds with geometrically consistent synthetic points by exploiting multi-view constraints and adaptive spatial sampling?

---

## Technical Contribution

### Core Innovation

This research introduces a **hybrid geometric-photometric augmentation framework** that combines:

1. **Adaptive spatial sampling** via quadtree decomposition that concentrates effort on information-rich image regions
2. **Local depth interpolation** through nearest-neighbor plane fitting in camera space
3. **Multi-view consensus voting** mechanism that verifies geometric consistency across multiple viewpoints

The key insight is that unsampled image regions (gaps in the quadtree occupancy map) can be assigned 3D positions by:
- Interpolating depth from nearby reconstructed points
- Validating these hypotheses across neighboring camera views
- Accepting only points that satisfy geometric constraints in multiple views

### Novelty vs. Existing Work

**vs. Multi-View Stereo (MVS):** Unlike dense MVS methods (COLMAP MVS, OpenMVS) that require photometric optimization and produce full depth maps, our approach targets *selective densification* of sparse clouds, operating directly on 3D points rather than per-pixel depths.

**vs. Interpolation Methods:** Unlike naive spatial interpolation (IDW, kriging), our method enforces multi-view geometric consistency through cross-projection verification, preventing hallucinated points in occluded or invalid regions.

**vs. Learning-Based Completion:** Unlike neural completion methods, our approach requires no training data and provides geometric guarantees through explicit epipolar consistency checks.

---

## Methodology

### 1. Quadtree-Based Spatial Decomposition

**Motivation:** Uniform sampling wastes computation in homogeneous regions where depth is locally predictable, while under-sampling textured regions where variation indicates geometric detail.

**Approach:** For each camera $C_i$, recursively partition image $I_i$ based on intensity variance:

$$
\text{subdivide}(N) \iff \sigma(I_i(\Omega_N)) > \tau_{std} \land |\Omega_N| > \tau_{min}
$$

where $\sigma(\cdot)$ denotes standard deviation, $\tau_{std}$ is the texture threshold (default: 7), and $\tau_{min}$ is the minimum node size (default: $5 \times 5$ pixels).

**Occupancy Classification:** Project existing point cloud $\mathcal{P}$ onto image plane. Leaf node $N_\ell$ is:
- **Occupied** if $\exists P_j \in \mathcal{P} : \pi(K_i, R_i, t_i, P_j) \in \Omega_\ell$
- **Unoccupied** otherwise (candidate for augmentation)

**Sampling Strategy:** For each unoccupied leaf node $N_\ell$, sample one candidate point $\mathbf{u}_s \sim \mathcal{U}(\Omega_\ell)$ uniformly within the node region, extracting its RGB color $\mathbf{c}_s = I_i(\mathbf{u}_s)$.

---

### 2. Depth Estimation via Local Plane Fitting

**Problem:** Given a 2D point $\mathbf{u}_s$ in an unoccupied node, estimate its 3D depth $z_s$.

**Approach:**

**Step 1: Nearest Neighbor Retrieval**

Find $k=3$ nearest neighbors $\\{P_{n_1}, P_{n_2}, P_{n_3}\\}$ in image space using BallTree spatial index with complexity $O(\log n)$.

**Step 2: Camera Space Transformation**

Transform points to camera coordinates:

$$
P_j^{cam} = R_i P_j + t_i
$$

**Step 3: Plane Fitting**

Fit a plane through the three neighbors in camera space. The plane normal is computed via cross product:

$$
\mathbf{n} = (P_{n_2}^{cam} - P_{n_1}^{cam}) \times (P_{n_3}^{cam} - P_{n_1}^{cam})
$$

The plane equation is $\mathbf{n}^T \mathbf{x} = d$ where $d = \mathbf{n}^T P_{n_1}^{cam}$.

**Step 4: Ray-Plane Intersection**

The camera ray through $\mathbf{u}_s$ is:

$$
\mathbf{r}(t) = t \cdot K_i^{-1} \begin{bmatrix} \mathbf{u}_s \\\\ 1 \end{bmatrix}
$$

Solving $\mathbf{n}^T \mathbf{r}(t) = d$ yields depth:

$$
z_s = \frac{d}{\mathbf{n}^T K_i^{-1}[\mathbf{u}_s^T, 1]^T}
$$

**Step 5: Validity Constraints**

Reject samples where:

$$
z_s < \tau_{depth} \quad \text{or} \quad |\cos(\angle(\mathbf{n}, \mathbf{r}))| < \tau_{cos}
$$

with $\tau_{depth} = 2.0$ and $\tau_{cos} = 0.01$, ensuring positive depth and non-grazing angles.

**Step 6: World Space Reconstruction**

Valid samples are back-projected to world coordinates:

$$
P_s = R_i^T (z_s \cdot K_i^{-1}[\mathbf{u}_s^T, 1]^T - t_i)
$$

**Limitations:** This approach assumes local surface planarity within the support region of the three nearest neighbors. It may fail at sharp geometric discontinuities, thin structures, or highly curved surfaces.

---

### 3. Multi-View Geometric Consistency

**Core Hypothesis:** A valid 3D point should project consistently across multiple views, exhibiting both geometric consistency (depth agreement) and optional photometric consistency (appearance similarity).

**Verification Protocol:**

For each candidate point $P_s$ from camera $C_i$:

**Step 1: Neighborhood Selection**

Define a verification set of $m=12$ neighboring cameras:

$$
\mathcal{V}_i = \\{C_{(i+k) \bmod N}\\}_{k \in \\{-6,\ldots,-1,1,\ldots,6\\}}
$$

This ensures circular wrapping for sequential camera orderings (e.g., video sequences).

**Step 2: Cross-View Projection**

Project each candidate into verification view $C_v \in \mathcal{V}_i$:

$$
\mathbf{u}_s^v = \pi(K_v, R_v, t_v, P_s), \quad z_s^v = (R_v P_s + t_v)_z
$$

**Step 3: Geometric Filtering**

Apply hard constraints to reject invalid projections:

$$
\text{valid}(\mathbf{u}_s^v) = \begin{cases}
\text{false} & \text{if } \mathbf{u}_s^v \notin [0,w] \times [0,h] \\\\
\text{false} & \text{if } z_s^v \leq 0 \\\\
\text{true} & \text{otherwise}
\end{cases}
$$

**Step 4: Depth Consistency Check**

Locate the quadtree leaf node $N_\ell^v$ at $\mathbf{u}_s^v$ in view $C_v$. The consistency criterion depends on occupancy status:

*For unoccupied nodes* (with sampled depth $z_\ell^v$):

$$
\text{consistent}_{depth} = |z_s^v - z_\ell^v| < \tau_{tol} \cdot z_\ell^v
$$

*For occupied nodes* (with nearest point $P_{nearest}$ at depth $z_{ref}^v$):

$$
\text{consistent}_{depth} = |z_s^v - z_{ref}^v| < \tau_{tol} \cdot z_{ref}^v
$$

where $\tau_{tol} = 0.2$ (20% relative tolerance).

**Step 5: Photometric Consistency (Optional)**

When enabled via `--compare_center_patch`, extract $3 \times 3$ patches $\mathcal{I}_s, \mathcal{I}_\ell$ centered at $\mathbf{u}_s^v$ and compare using Gaussian-weighted L2 distance:

$$
d_{photo} = \sum_{\mathbf{p} \in \mathcal{N}_{3 \times 3}} w(\mathbf{p}) \|\mathcal{I}_s(\mathbf{p}) - \mathcal{I}_\ell(\mathbf{p})\|_2
$$

where $w(\mathbf{p})$ is a 2D Gaussian kernel. Accept if $d_{photo} < \tau_{patch} = 0.5$.

**Step 6: Consensus Voting**

For each candidate $P_s$, track:

$$
n_{infer} = \sum_{C_v \in \mathcal{V}_i} \mathbb{1}[\text{valid}(\mathbf{u}_s^v)]
$$

$$
n_{reject} = \sum_{C_v \in \mathcal{V}_i} \mathbb{1}[\text{valid}(\mathbf{u}_s^v) \land \neg \text{consistent}(\mathbf{u}_s^v)]
$$

**Acceptance Criterion:**

A point is retained if:

$$
n_{infer} - n_{reject} \geq \tau_{votes} = 1
$$

This ensures at least one net positive verification, balancing precision and recall by requiring more consistent observations than rejections.

---

### 4. Point Cloud Fusion

Accepted points $\mathcal{P}_{aug}$ are merged with the original reconstruction $\mathcal{P}$ and exported in COLMAP's binary format (`points3D.bin`). Each augmented point stores:

$$
P_{aug} = \\{\mathbf{x} \in \mathbb{R}^3, \mathbf{c} \in [0,255]^3, e_{repr} = 0, \text{track} = \\{(C_i, \mathbf{u}_s)\\}\\}
$$

where:
- $\mathbf{x}$: 3D world coordinates
- $\mathbf{c}$: RGB color values
- $e_{repr}$: Reprojection error (set to 0 for synthetic points)
- $\text{track}$: Single observation from source camera with dummy indices

---

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quadtree_std_threshold` | 7 | Texture variation threshold for subdivision |
| `quadtree_min_pixel_size` | 5 | Minimum node size in pixels |
| `depth_cutoff` | 2.0 | Minimum valid depth value |
| `cosine_threshold` | 0.01 | Minimum angle between plane normal and ray |
| `depth_tolerance` | 0.2 | Relative depth difference threshold (20%) |
| `patch_threshold` | 0.5 | Appearance similarity threshold |
| `consistency_votes` | 1 | Minimum net positive votes required |
| `n_neighbors_verification` | 12 | Number of views for multi-view checks |

---

## Optional Features

### Visibility-Aware Culling

**Flag:** `--visibility_aware_culling`

**Purpose:** Removes projected 3D points whose observed color differs significantly from their stored RGB values, helping eliminate occluded or incorrectly matched points.

**Mechanism:** Uses pixelwise RGB difference threshold of 0.3 (normalized to [0,1]). Points failing this test are culled before occupancy classification.

### Patch Comparison

**Flag:** `--compare_center_patch`

**Purpose:** Enables local texture matching using $3 \times 3$ patches for additional photometric consistency beyond depth checks.

**Mechanism:** Computes Gaussian-weighted L2 distance between patches in source and verification views.

### Camera Ordering

**Flag:** `--camera_order`

**Options:**
- `covisibility` (default): Process cameras based on shared 3D point observations
- `sequential`: Process by camera ID order

**Purpose:** Determines the sequence for processing views and selecting neighboring verification cameras.

### Clustering

**Flag:** `--n_clusters`

**Default:** 10

**Purpose:** Groups cameras into spatial clusters for large datasets, enabling localized processing within camera neighborhoods.

---

## Computational Complexity

### Time Complexity

- **Quadtree construction:** $O(n_{pixels} \log n_{pixels})$ per image
- **Nearest neighbor search:** $O(n_{candidates} \log n_{sparse})$ via BallTree
- **Multi-view verification:** $O(n_{candidates} \times m_{neighbors})$
- **Overall per image:** $O(n_{pixels} \log n_{pixels} + n_{candidates}(m_{neighbors} + \log n_{sparse}))$

### Space Complexity

- **Quadtree storage:** $O(n_{pixels} / \tau_{min})$ nodes
- **BallTree index:** $O(n_{sparse})$
- **Overall:** $O(n_{pixels} / \tau_{min} + n_{sparse})$

---

## Limitations and Failure Cases

### Inherent Assumptions

1. **Local Planarity:** The 3-point plane fitting assumes surfaces are locally planar within the nearest neighbor support region. This fails at:
   - Sharp geometric discontinuities (edges, corners)
   - Highly curved surfaces (spheres, cylinders at small scales)
   - Thin structures where foreground and background are confused

2. **Lambertian Surfaces:** Photometric consistency assumes diffuse reflectance. Fails on:
   - Specular surfaces (glass, metal, water)
   - View-dependent materials
   - Surfaces with strong BRDF variations

3. **Static Scenes:** Multi-view consistency requires temporally stable geometry. Cannot handle:
   - Dynamic objects
   - Moving elements between captures
   - Deformable surfaces

4. **Sufficient Overlap:** Requires multiple overlapping views for verification. Limited effectiveness in:
   - Boundary regions with few neighbors
   - Low-overlap datasets
   - Sparse camera distributions

### Known Failure Modes

**Thin Structures:** Planes may be fit through points spanning foreground and background, generating invalid depths near thin objects (wires, branches, railings).

**Repetitive Textures:** Quadtree may over-subdivide periodic patterns, increasing computational cost without improving geometric accuracy.

**Textureless Regions:** In homogeneous areas, quadtree may not subdivide sufficiently, missing valid augmentation opportunities.

**Occlusions:** While depth comparison rejects many occluded points, ambiguous cases near occlusion boundaries may still produce false positives.

---

## Usage Example

Basic augmentation workflow:

```bash
python augment.py \
  --colmap_path /path/to/colmap/sparse/0 \
  --image_path /path/to/images \
  --augment_path /path/to/output/points3D.bin \
  --camera_order covisibility \
  --n_clusters 10
```

With optional features enabled:

```bash
python augment.py \
  --colmap_path /path/to/colmap/sparse/0 \
  --image_path /path/to/images \
  --augment_path /path/to/output/points3D.bin \
  --camera_order covisibility \
  --n_clusters 10 \
  --visibility_aware_culling \
  --compare_center_patch
```

---

## Implementation Details

### Dependencies

- Python 3.8+
- NumPy (array operations)
- SciPy (BallTree spatial index)
- OpenCV (image I/O, projection operations)
- COLMAP Python bindings (point cloud I/O in binary format)

### Key Data Structures

**Quadtree Node:**
```python
class QuadtreeNode:
    bbox: tuple          # (x, y, width, height)
    std_dev: float       # Intensity standard deviation
    is_leaf: bool        # Leaf node flag
    is_occupied: bool    # Contains projected 3D points
    sampled_point: tuple # (u, v, rgb) if unoccupied
    children: list       # Child nodes if subdivided
```

**Candidate Point:**
```python
class CandidatePoint:
    xyz_world: np.ndarray     # 3D world coordinates
    rgb: np.ndarray           # RGB color [0, 255]
    source_camera: int        # Camera ID where sampled
    pixel_coord: np.ndarray   # Source 2D coordinates
    n_infer: int              # Valid projections count
    n_reject: int             # Rejected projections count
```

---

## Future Directions

### Algorithmic Improvements

1. **Higher-Order Surface Fitting:** Extend beyond planar assumption using quadric surfaces or local polynomial fitting for curved geometry.

2. **Adaptive Neighbor Selection:** Dynamically select verification views based on baseline geometry and overlap rather than sequential ordering.

3. **Multi-Scale Quadtree:** Implement hierarchical consistency checks across multiple quadtree levels for improved robustness.

4. **Uncertainty Quantification:** Derive per-point confidence scores from consensus statistics and geometric quality metrics.

### Integration Opportunities

1. **Neural Depth Priors:** Combine with learned depth estimation for improved handling of textureless regions while maintaining geometric guarantees.

2. **Semantic Guidance:** Leverage semantic segmentation to adapt sampling strategy per object category (e.g., finer sampling on small objects).

3. **Mesh-Aware Processing:** Generate augmented points directly on surface meshes reconstructed from sparse clouds.

4. **GPU Acceleration:** Parallelize quadtree construction, projection, and consistency checks using CUDA or OpenCL.

---

## Research Context

**Keywords:** Structure-from-Motion, Point Cloud Augmentation, Multi-View Geometry, Spatial Data Structures, 3D Reconstruction, Geometric Consistency, Quadtree Decomposition

**Related Work:**
- Multi-View Stereo reconstruction
- Point cloud completion
- Depth map fusion
- Geometric consistency checking
- Adaptive spatial sampling

**Potential Applications:**
- SfM post-processing pipelines
- Initialization for dense MVS methods
- 3D scene understanding
- Robotic mapping and navigation
- Cultural heritage documentation

---

## Conclusion

This research presents a method for selectively densifying sparse Structure-from-Motion reconstructions through adaptive spatial sampling and multi-view geometric consistency verification. The approach fills a methodological gap between sparse feature-based reconstruction and dense photometric methods by:

1. Concentrating computational effort on geometrically informative regions via quadtree decomposition
2. Providing geometric guarantees through explicit multi-view consensus voting
3. Operating directly on 3D point clouds without requiring dense depth map estimation

The method is designed as a practical post-processing step for existing SfM pipelines, requiring no training data while providing interpretable geometric validation. Key limitations include the local planarity assumption and dependency on sufficient multi-view overlap, suggesting future integration with learning-based approaches for handling challenging geometric configurations.
