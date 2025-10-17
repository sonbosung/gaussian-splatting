# Contributions to 3D Gaussian Splatting

This document outlines the key contributions and enhancements made to the original 3D Gaussian Splatting implementation from [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

## Overview

This implementation extends the original 3D Gaussian Splatting framework with several novel training strategies and point cloud augmentation techniques aimed at improving reconstruction quality, training efficiency, and handling sparse input data. The contributions can be categorized into three main areas: **Bundle-Based Training**, **Advanced Loss Functions**, and **Point Cloud Augmentation**.

---

## 1. Bundle-Based Training Strategy

### Motivation
Traditional 3D Gaussian Splatting randomly samples training views, which can lead to inefficient gradient propagation and slower convergence, especially during the critical densification phase. By leveraging camera covisibility information, we can guide the training process to focus on spatially coherent regions.

### Implementation (`train.py` Lines 86-96, 128-158)

**Camera Ordering System:**
- Implements three camera ordering strategies via `utils/bundle_utils.py`:
  - **Covisibility-based ordering**: Constructs a graph based on shared 3D points between camera pairs and performs greedy traversal to maximize consecutive overlap
  - **PCA-based ordering**: Projects camera centers onto a 2D plane and orders by angular position around the centroid
  - **Clustering-based ordering**: Groups cameras into spatial clusters using k-means

**Focused Sampling During Densification:**
```python
if bundle_training and iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % 100 >= 80:
    # Sample from camera cluster (20 consecutive cameras in ordered sequence)
    group_uid_stack = ordered_uids[start_idx:start_idx + 20]
```

During iterations 80-99 of each 100-iteration cycle within the densification period (default: iterations 500-15000), the system samples exclusively from a cluster of 20 consecutive cameras in the ordered sequence. This concentrates gradient statistics in specific scene regions, enabling more effective densification decisions.

**Cluster Progression:**
After each densification operation (every 100 iterations), the system advances to the next camera cluster via `n_interval += 1`, systematically ensuring all scene regions receive focused attention during point cloud refinement.

### Key Command-Line Arguments
- `--bundle_training`: Enable bundle-based training mode
- `--camera_order [covisibility|pca|cluster]`: Select camera ordering strategy (default: covisibility)

### Benefits
- **Improved convergence**: Consecutive views share visual features, providing coherent gradients
- **Better densification**: Focused sampling leads to more accurate gradient statistics for splitting/cloning decisions
- **Scene-aware training**: Respects the geometric structure of camera arrangements

---

## 2. Advanced Loss Functions and Regularization

### 2.1 Inverse Depth Smoothness Loss

**Motivation**: Rendered depth maps can contain artifacts or discontinuities in regions with insufficient geometric constraints. A smoothness regularizer helps produce more coherent depth maps while preserving edges.

**Implementation** (`utils/loss_utils.py`):
```python
class InvDepthSmoothnessLoss(nn.Module):
    def forward(self, depth, image):
        # Computes edge-aware smoothness using image gradients as weights
        # Penalizes depth discontinuities in non-edge regions
```

Applies edge-aware smoothness to inverse depth predictions, using image gradients to avoid penalizing valid depth discontinuities at object boundaries.

### 2.2 Laplacian Pyramid Loss

**Motivation**: Multi-scale texture comparison captures both fine details and broader structural similarities, complementing the single-scale L1 and SSIM losses.

**Implementation** (`utils/loss_utils.py`):
```python
def laplacian_pyramid_loss(pred, target, num_levels=4):
    # Builds 4-level Laplacian pyramids for both images
    # Computes L1 loss at each pyramid level
```

Constructs Laplacian pyramids (default: 4 levels) for both rendered and ground truth images, computing L1 loss across scales to capture multi-resolution texture fidelity.

### Combined Loss Function (`train.py` Lines 181-188)

```python
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + lambda_ds * ds_loss + lambda_lap * lap_loss
```

The total loss balances:
- **L1 Loss** (80% default): Pixel-wise accuracy
- **SSIM Loss** (20% default): Perceptual quality
- **Depth Smoothness** (optional, weighted by `lambda_ds`): Geometric coherence
- **Laplacian Pyramid** (optional, weighted by `lambda_lap`): Multi-scale texture matching

### Key Command-Line Arguments
- `--enable_ds_lap`: Enable depth smoothness and Laplacian pyramid losses
- `--lambda_ds [float]`: Weight for depth smoothness loss (default: 0.0)
- `--lambda_lap [float]`: Weight for Laplacian pyramid loss (default: 0.0)

---

## 3. Point Cloud Augmentation System

### Motivation
COLMAP structure-from-motion often produces sparse point clouds, particularly in:
- Textureless regions (walls, floors)
- Specular or reflective surfaces
- Areas with limited view overlap

Sparse initialization leads to incomplete scene coverage and requires aggressive densification during training. The augmentation system addresses this by intelligently generating synthetic 3D points before training begins.

### Core Approach (`augment.py` + `AUGMENTATION_METHOD.md`)

The augmentation pipeline consists of four phases:

#### Phase 1: Quadtree-Based Adaptive Sampling
- **Quadtree Decomposition**: Recursively subdivides each image based on pixel intensity variation (standard deviation)
  - Subdivide if: `std_dev > threshold` (default: 7) and `size > min_size` (default: 5×5 pixels)
  - Creates adaptive sampling density: dense in textured regions, sparse in smooth areas

- **Unoccupied Region Detection**: Project existing COLMAP 3D points onto image plane
  - Mark leaf nodes as "occupied" (contains projected points) or "unoccupied" (candidate for augmentation)
  - Sample one random pixel per unoccupied node

#### Phase 2: Depth Interpolation
For each sampled 2D point:
1. **Nearest Neighbor Search**: Find 3 nearest projected 3D points using BallTree
2. **Local Plane Fitting**: Construct a plane through the 3 neighbors in camera space
3. **Ray-Plane Intersection**: Compute depth by intersecting camera ray with fitted plane
4. **Validation**: Reject if:
   - Depth is negative or too small (< 2.0 units)
   - Plane is nearly parallel to ray (angle < 0.01 radians)
   - Computation produces NaN values

#### Phase 3: Multi-View Consistency Verification
Each candidate point must be geometrically consistent across neighboring views:

1. **Neighbor Selection**: Check consistency with 12 neighboring cameras (±1 to ±6 in ordered sequence)
2. **Projection and Matching**: Project point into neighbor view and find corresponding quadtree node
3. **Consistency Checks**:
   - **Depth consistency**: `|depth_A - depth_B| < 0.2 × depth_B` (20% tolerance)
   - **Appearance consistency** (optional): Compare 3×3 RGB patches using Gaussian-weighted L2 distance
4. **Voting**: Accept point if `(valid_projections - rejections) ≥ 1`

This ensures only geometrically plausible points that are visible and consistent across multiple views are added.

#### Phase 4: Output Generation
- Append validated synthetic points to COLMAP's `points3D.bin` format
- Preserve original COLMAP points unchanged
- Assign synthetic points minimal track length (1 observation) and zero reprojection error

### Key Parameters and Utilities

**Augmentation System** (`augment.py`):
- `--colmap_path`: Path to COLMAP sparse reconstruction
- `--image_path`: Path to input images
- `--augment_path`: Output path for augmented points3D.bin
- `--camera_order [covisibility|pca|cluster]`: Camera ordering for consistency checks
- `--n_clusters [int]`: Number of camera clusters (default: 10)
- `--visibility_aware_culling`: Enable occlusion-aware point culling
- `--compare_center_patch`: Enable patch-based appearance matching

**Supporting Utilities**:
- `utils/bundle_utils.py`: Camera ordering and clustering algorithms
- `utils/aug_utils.py`: Quadtree construction, depth interpolation, consistency checking
- `utils/colmap_utils.py`: COLMAP data reading and camera parameter extraction

### Augmentation Workflow
```bash
# 1. Run COLMAP to get initial sparse reconstruction
# 2. Augment the point cloud
python augment.py \
  --colmap_path /path/to/sparse/0 \
  --image_path /path/to/images \
  --augment_path /path/to/output/points3D.bin \
  --camera_order covisibility \
  --visibility_aware_culling

# 3. Train with augmented point cloud
python train.py -s /path/to/scene --bundle_training --camera_order covisibility
```

### Benefits
- **Denser Initialization**: Reduces reliance on aggressive densification during training
- **Better Coverage**: Fills in textureless regions that COLMAP misses
- **Geometric Consistency**: Multi-view verification ensures plausible point locations
- **Adaptive Density**: Quadtree focuses augmentation on under-sampled areas

---

## 4. Enhanced Monitoring and Logging

### Real-Time Evaluation (`train.py` Lines 219-225)
```python
if iteration % 1000 == 0:
    psnr_test, ssim_test = evaluate_test_images(scene, render, ...)
    print(f"[ITER {iteration}] Test PSNR: {psnr_test:.4f}, Test SSIM: {ssim_test:.4f}")
```

Evaluates all test images every 1000 iterations, providing:
- Real-time tracking of reconstruction quality
- Early detection of overfitting or training issues
- Quantitative metrics without waiting for final evaluation

### Training Logs (`train.py` Lines 281-290)
Automatically saves training metrics:
- `psnr_log.txt`: PSNR progression throughout training
- `ssim_log.txt`: SSIM progression throughout training
- `training_args.log`: Complete record of all command-line arguments

### Enhanced TensorBoard Logging
Additional loss components logged when `--enable_ds_lap` is active:
- `train_loss_patches/ssim_loss`: Structural similarity component
- `train_loss_patches/ds_loss`: Depth smoothness regularization
- `train_loss_patches/lap_loss`: Laplacian pyramid loss
- `train_loss_patches/lambda_ds`: Depth smoothness weight
- `train_loss_patches/lambda_lap`: Laplacian pyramid weight

---

## 5. Additional Training Variants

The repository includes multiple training script variations for experimental purposes:

- **`train_scheduler.py`**: Implements learning rate scheduling strategies
- **`train_warmup.py`**: Gradual warm-up phases for optimization stability
- **`train_partialscheduler.py`**: Selective scheduling for specific parameter groups
- **`train_newaug.py`**: Alternative augmentation strategies during training
- **`train_colmap_anis.py`**: Specialized initialization from COLMAP anisotropic data
- **`train_directls.py`**: Direct least-squares optimization variants
- **`train_basiccluster.py`**: Simplified clustering-based training

These variants explore different optimization strategies, initialization methods, and augmentation techniques beyond the main `train.py` implementation.

---

## 6. Initial Model Saving

### Implementation (`train.py` Line 100)
```python
scene.save(0)
```

Saves the initial Gaussian model state (iteration 0) before training begins. This enables:
- Direct comparison of initialization quality
- Debugging initialization issues
- Analysis of how much improvement comes from optimization vs. initialization

---

## Summary of Key Contributions

| Contribution | Files Modified/Added | Impact |
|-------------|---------------------|--------|
| **Bundle-Based Training** | `train.py`, `utils/bundle_utils.py` | Improved convergence and densification through camera-aware sampling |
| **Depth Smoothness Loss** | `train.py`, `utils/loss_utils.py` | Enhanced geometric coherence in rendered depth maps |
| **Laplacian Pyramid Loss** | `train.py`, `utils/loss_utils.py` | Multi-scale texture fidelity for better detail preservation |
| **Point Cloud Augmentation** | `augment.py`, `utils/aug_utils.py`, `utils/colmap_utils.py` | Denser initialization for sparse COLMAP reconstructions |
| **Real-Time Evaluation** | `train.py` | Continuous quality monitoring during training |
| **Enhanced Logging** | `train.py` | Comprehensive metric tracking and reproducibility |

---

## Usage Examples

### Basic Training with Bundle Strategy
```bash
python train.py -s /path/to/scene \
  --bundle_training \
  --camera_order covisibility
```

### Training with Advanced Loss Functions
```bash
python train.py -s /path/to/scene \
  --bundle_training \
  --enable_ds_lap \
  --lambda_ds 0.01 \
  --lambda_lap 0.1
```

### Full Pipeline with Augmentation
```bash
# Step 1: Augment point cloud
python augment.py \
  --colmap_path /path/to/sparse/0 \
  --image_path /path/to/images \
  --augment_path /path/to/augmented/points3D.bin \
  --camera_order covisibility \
  --visibility_aware_culling

# Step 2: Train with augmented data
python train.py -s /path/to/scene \
  --bundle_training \
  --camera_order covisibility \
  --enable_ds_lap \
  --lambda_ds 0.01 \
  --lambda_lap 0.05
```

---

## Technical References

For detailed technical documentation:
- **Augmentation Algorithm**: See `AUGMENTATION_METHOD.md` for complete algorithmic details
- **Training Pipeline**: See `TECHNICAL_OVERVIEW.md` for in-depth architecture documentation
- **Original 3DGS Paper**: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

---

## Acknowledgments

This work builds upon the original 3D Gaussian Splatting implementation by Kerbl et al. (SIGGRAPH 2023). The enhancements focus on improving training efficiency and handling challenging sparse reconstruction scenarios while maintaining the real-time rendering capabilities of the original method.

**Original 3DGS Citation:**
```bibtex
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
