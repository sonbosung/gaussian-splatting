# 3D Point Cloud Augmentation Method

## Overview

This augmentation method generates new 3D points to densify sparse point clouds reconstructed from COLMAP. The method uses quadtree-based spatial decomposition to identify unsampled image regions and interpolates their 3D locations through multi-view geometric consistency checks.

## Core Approach

The method operates in three main phases:

### Phase 1: Quadtree Decomposition and Sampling

For each input camera view:

1. **Build Quadtree**: Recursively subdivide the image into regions based on pixel intensity variation (standard deviation). Regions with high variation are subdivided further until either:
   - The region's standard deviation falls below a threshold (default: 7)
   - The region size reaches a minimum pixel size (default: 5×5)

2. **Identify Unoccupied Regions**: Project existing 3D points onto the image plane. Mark quadtree leaf nodes as:
   - **Occupied**: Contains at least one projected 3D point
   - **Unoccupied**: Contains no projected 3D points (candidate for augmentation)

3. **Sample New Points**: For each unoccupied leaf node:
   - Sample a random pixel location within the node
   - Extract RGB color at that location
   - Store the 2D pixel coordinate for depth estimation

### Phase 2: Depth Interpolation

For unoccupied nodes, estimate 3D depth using nearest neighbor interpolation:

1. **Find Nearest 3D Points**: For each sampled 2D point, find the 3 nearest projected 3D points (using BallTree search)

2. **Fit Local Plane**: Construct a plane through the 3 nearest neighbors in camera space

3. **Compute Depth**: Intersect the camera ray (passing through the sampled pixel) with the fitted plane to determine depth

4. **Validate Depth**: Reject samples where:
   - Depth is negative or too small (< 2.0 units)
   - The fitted plane is nearly parallel to the camera ray (angle < 0.01 radians)
   - Depth computation produces NaN values

5. **Transform to World Space**: Convert valid (pixel, depth) pairs to 3D world coordinates

### Phase 3: Multi-View Consistency Verification

To ensure geometric validity, each sampled point is verified across multiple neighboring views:

1. **Select Neighboring Views**: For each view, check consistency with 12 neighboring views in the camera ordering (±1 to ±6 cameras away, wrapping circularly)

2. **Project and Match**: For each sampled point from view A projected into view B:
   
   **a) Geometric Filtering**:
   - **Cull** if point projects outside image boundaries
   - **Cull** if point has negative depth in view B
   
   **b) Locate Corresponding Node**: Find the quadtree leaf node in view B at the projected location
   
   **c) Consistency Checks**:
   
   If the corresponding node is **unoccupied** (sampled):
   - Compare depths: Accept if `|depth_A - depth_B| < 0.2 × depth_B`
   - Optionally compare 3×3 RGB patches using Gaussian-weighted difference (threshold: 0.5)
   
   If the corresponding node is **occupied** (contains 3D points):
   - Compare with nearest 3D point depth: Accept if error < 20% of reference depth
   - Optionally compare RGB patches with the matching 3D point

3. **Track Consistency**:
   - `inference_count`: Number of valid projections (not culled or missing)
   - `rejection_count`: Number of projections rejected due to depth/appearance mismatch

4. **Acceptance Criterion**: A sampled point is accepted if:
   ```
   inference_count - rejection_count ≥ 1
   ```
   This means the point must be geometrically consistent in at least one more view than it is rejected.

### Phase 4: Output Generation

1. **Filter Valid Points**: Select all sampled points that pass the multi-view consistency threshold

2. **Write Augmented Point Cloud**: Append the new points to COLMAP's `points3D.bin` format with:
   - 3D world coordinates (xyz)
   - RGB color values
   - Zero reprojection error (these are synthetic points)
   - Minimal track length (1 observation with dummy indices)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quadtree_std_threshold` | 7 | Maximum standard deviation before subdivision |
| `quadtree_min_pixel_size` | 5 | Minimum node size in pixels |
| `depth_cutoff` | 2.0 | Minimum valid depth value |
| `cosine_threshold` | 0.01 | Maximum angle between plane normal and ray |
| `depth_tolerance` | 0.2 | Relative depth difference threshold (20%) |
| `patch_threshold` | 0.5 | Appearance similarity threshold (if enabled) |
| `consistency_votes` | 1 | Minimum net positive votes needed |

## Optional Features

**Visibility-Aware Culling** (`--visibility_aware_culling`):
- Culls projected 3D points whose observed color differs significantly from stored RGB
- Helps remove occluded or incorrectly matched points
- Uses pixelwise RGB difference threshold of 0.3 (normalized to [0,1])

**Patch Comparison** (`--compare_center_patch`):
- Enables local texture matching using 3×3 patches
- Uses Gaussian-weighted L2 distance between patches
- Provides additional photometric consistency beyond depth checks

**Camera Ordering** (`--camera_order`):
- Determines the sequence for processing views
- Default: "covisibility" (processes cameras based on shared observations)
- Alternative: sequential ordering by camera ID

**Clustering** (`--n_clusters`):
- Groups cameras into spatial clusters (default: 10)
- Enables processing large datasets by focusing on local camera neighborhoods

## Algorithm Strengths

1. **Adaptive Sampling**: Quadtree focuses sampling on textured regions while avoiding over-sampling smooth areas
2. **Geometric Consistency**: Multi-view verification ensures new points are geometrically plausible
3. **Occlusion Handling**: Depth comparison rejects points that would be occluded in other views
4. **Density Control**: Per-node sampling prevents over-densification

## Limitations

1. **Planar Assumption**: Depth interpolation assumes local planarity (3-point plane fitting)
2. **Texture Dependency**: Requires sufficient texture for quadtree subdivision
3. **View Coverage**: Requires multiple overlapping views for consistency verification
4. **Computational Cost**: Quadtree construction and multi-view projection can be expensive for large datasets

## Usage Example

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

## Output

The method outputs an augmented `points3D.bin` file containing:
- All original COLMAP points (preserved exactly)
- New synthetic points that passed multi-view consistency checks

The console displays:
- Total number of sampled points across all views
- Final point count after consistency filtering
- Last assigned point ID in the augmented cloud
