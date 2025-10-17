# Technical Project Overview: train.py

## Overview

The `train.py` module implements a 3D Gaussian Splatting training pipeline for novel view synthesis from multi-view images. This implementation extends the base Gaussian Splatting approach with advanced features including bundle-based training, depth regularization, exposure optimization, and optional image processing losses.

The system represents 3D scenes as collections of anisotropic 3D Gaussians that are optimized through differentiable rendering. Each Gaussian is characterized by position, opacity, color (via spherical harmonics), scale, and rotation parameters that are jointly optimized to minimize photometric reconstruction loss.

### Key Capabilities

- **Differentiable Gaussian Rasterization**: Real-time rendering of 3D Gaussians with gradient computation for optimization
- **Adaptive Densification**: Dynamic point cloud refinement through cloning and splitting based on gradient statistics
- **Bundle Training**: Camera-order-aware training using covisibility or spatial clustering for improved convergence
- **Depth Regularization**: Optional monocular depth supervision with adaptive weighting
- **Advanced Loss Functions**: L1, SSIM, inverse depth smoothness, and Laplacian pyramid losses
- **Exposure Compensation**: Per-camera exposure parameters for handling lighting variations
- **Sparse Adam Optimization**: Optional memory-efficient optimization for large scenes

---

## Core Flow

### 1. Initialization Phase (Lines 60-100)

The training begins by initializing the scene representation and configuring the optimization pipeline:

**Scene Construction**
```python
gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
scene = Scene(dataset, gaussians)
gaussians.training_setup(opt)
```

The `Scene` class loads COLMAP reconstruction data and initializes the Gaussian point cloud from the structure-from-motion output. The `GaussianModel` encapsulates all learnable parameters including:
- `_xyz`: 3D positions
- `_features_dc` and `_features_rest`: Spherical harmonic coefficients for view-dependent color
- `_scaling`: Logarithmic scale parameters
- `_rotation`: Quaternion rotations
- `_opacity`: Inverse sigmoid opacity values

**Camera Organization** (Lines 88-96)

When `bundle_training` is enabled, cameras are reordered using one of three strategies:

1. **Covisibility-based** (`utils/bundle_utils.py:cluster_cameras`): Constructs a graph where edges represent shared 3D points between camera pairs, then performs greedy traversal to maximize consecutive covisibility
2. **PCA-based**: Projects camera centers onto 2D plane via PCA and orders by angular position
3. **Clustering-based**: Groups cameras into spatial clusters using k-means on camera positions

This ordering improves gradient flow during early training by ensuring consecutive views share visual content.

### 2. Main Training Loop (Lines 104-280)

The training proceeds through iterations with the following structure:

#### Camera Selection (Lines 128-158)

**Standard Mode**: Random sampling from training cameras
```python
rand_idx = randint(0, len(viewpoint_stack) - 1)
viewpoint_cam = viewpoint_stack.pop(rand_idx)
```

**Bundle Mode** (iterations 80-99 every 100 steps): Focuses on camera clusters during densification
```python
start_idx = start_indices[n_interval] % len(ordered_uids)
group_uid_stack = ordered_uids[start_idx:start_idx + 20]
```

This focused sampling concentrates gradient statistics in specific scene regions, enabling more effective densification.

#### Rendering Pass (Lines 166-167)

```python
render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                    use_trained_exp=dataset.train_test_exp, 
                    separate_sh=SPARSE_ADAM_AVAILABLE)
```

The `gaussian_renderer/render` function performs:
1. Frustum culling of Gaussians outside view
2. Screen space projection of 3D Gaussians
3. Alpha-blending in sorted order (front-to-back)
4. Accumulation of color and depth

Output includes rendered image, viewspace gradients, visibility masks, and radii for densification tracking.

#### Loss Computation (Lines 174-203)

**Photometric Loss** (Lines 174-188):
```python
Ll1 = l1_loss(image, gt_image)
ssim_value = ssim(image, gt_image)
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
```

Default weighting: 80% L1 + 20% SSIM (via `lambda_dssim=0.2`), balancing pixel-wise accuracy with perceptual quality.

**Optional Regularization** (Lines 181-186):
- `InvDepthSmoothnessLoss`: Penalizes depth discontinuities at non-edge regions using image gradient weighting
- `laplacian_pyramid_loss`: Multi-scale texture comparison across 4 pyramid levels

**Depth Supervision** (Lines 191-202):

When monocular depth maps are available (`depth_reliable=True`):
```python
Ll1depth = depth_l1_weight(iteration) * torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
```

The weight decays exponentially from `depth_l1_weight_init` to `depth_l1_weight_final` via `utils/general_utils.py:get_expon_lr_func`, enabling strong early guidance that fades as photometric optimization improves.

#### Backward Pass and Densification (Lines 204-263)

```python
loss.backward()
```

Gradients flow through:
1. Differentiable rasterizer (custom CUDA kernels)
2. 3D Gaussian projection and covariance computation
3. Learnable parameters (positions, scales, rotations, colors, opacities)

**Densification Statistics** (Lines 253-254):
```python
gaussians.max_radii2D[visibility_filter] = torch.max(...)
gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
```

Tracks per-Gaussian gradient magnitudes and maximum screen-space radii for adaptive refinement.

**Densification Operations** (Lines 256-263):

Every `densification_interval` iterations (default 100) between iterations 500-15000:

```python
gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 
                             scene.cameras_extent, size_threshold, radii)
```

This performs (`scene/gaussian_model.py`):
- **Clone**: Duplicates under-reconstructed Gaussians (high gradient, small size)
- **Split**: Divides large Gaussians (high gradient, large size) into two smaller ones
- **Prune**: Removes transparent Gaussians (opacity < 0.005) or oversized ones

**Opacity Reset** (Line 262-263):

Every `opacity_reset_interval` iterations (default 3000):
```python
gaussians.reset_opacity()
```

Resets all opacities toward 0.01 to prevent optimization stagnation and allow pruning of redundant Gaussians.

#### Optimizer Step (Lines 266-275)

```python
gaussians.exposure_optimizer.step()  # Update per-camera exposure
if use_sparse_adam:
    gaussians.optimizer.step(visible, radii.shape[0])  # Sparse update
else:
    gaussians.optimizer.step()  # Dense Adam
```

The sparse Adam variant (`SparseGaussianAdam`) only updates parameters for visible Gaussians, significantly reducing memory for large scenes.

### 3. Evaluation and Checkpointing

**Periodic Testing** (Lines 219-225):

Every 1000 iterations, renders all test cameras:
```python
psnr_test, ssim_test = evaluate_test_images(scene, render, renderArgs)
```

**Checkpointing** (Lines 246-248, 277-279):

Saves model state at specified iterations:
```python
scene.save(iteration)  # Saves point cloud as PLY
torch.save((gaussians.capture(), iteration), checkpoint_path)  # Full state
```

---

## Key Functions

### training() (Lines 45-282)

**Purpose**: Main training orchestration  
**Parameters**:
- `dataset`: Model configuration (paths, camera params)
- `opt`: Optimization hyperparameters (learning rates, densification thresholds)
- `pipe`: Pipeline settings (SH computation mode, debugging flags)
- `testing_iterations`: When to evaluate on test set
- `saving_iterations`: When to save model checkpoints
- `checkpoint`: Path to resume from previous training
- `bundle_training`: Enable camera-order-aware training
- `enable_ds_lap`: Enable depth smoothness and Laplacian losses
- `lambda_ds`, `lambda_lap`: Loss weighting factors

**Key Operations**:
1. Initialize scene and Gaussian model
2. Setup optimizers (Adam for Gaussians, separate for exposure)
3. Order cameras if bundle training enabled
4. Loop over iterations:
   - Select camera (bundle-aware or random)
   - Render from camera viewpoint
   - Compute photometric + regularization losses
   - Backward pass
   - Update densification statistics
   - Perform densification/pruning (if in range)
   - Optimizer step
   - Periodic evaluation and saving
5. Log final metrics

**Returns**: None (modifies `gaussians` and `scene` in-place)

---

### prepare_output_and_logger() (Lines 292-312)

**Purpose**: Create output directory and initialize TensorBoard logging  
**Parameters**: `args` - Command line arguments with `model_path`

**Behavior**:
- Generates unique output path if none provided (using OAR job ID or UUID)
- Creates directory structure
- Saves configuration arguments
- Initializes TensorBoard `SummaryWriter` if available

**Returns**: `tb_writer` - TensorBoard writer or None

---

### training_report() (Lines 314-374)

**Purpose**: Log training progress and evaluate on validation sets  
**Parameters**:
- `tb_writer`: TensorBoard writer
- `iteration`: Current iteration number
- `Ll1`, `loss`, `ssim_loss`, `ds_loss`, `lap_loss`: Loss components
- `scene`, `renderFunc`, `renderArgs`: For validation rendering
- Various configuration flags

**Validation Process** (Lines 343-369):

Evaluates on two sets:
1. **Test cameras**: Full held-out views (1/8 of total)
2. **Train samples**: Every 5th training camera (indices 5, 10, 15, 20, 25)

For each camera:
- Renders image via `renderFunc`
- Computes L1 and PSNR metrics
- Logs first 5 renders as images to TensorBoard
- Averages metrics across set

Also logs:
- Opacity histogram (helps monitor densification)
- Total point count (tracks scene complexity)

---

### evaluate_test_images() (Lines 376-392)

**Purpose**: Efficiently compute average PSNR and SSIM on test set  
**Parameters**:
- `scene`: Contains test camera list
- `render_func`: Rendering function
- `render_args`: Packed arguments (pipe, background, etc.)

**Implementation**:
```python
for camera in test_cameras:
    rendered_image = torch.clamp(render_func(camera, scene.gaussians, *render_args)["render"], 0.0, 1.0)
    gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
    psnr_test += psnr(rendered_image, gt_image).mean()
    ssim_test += ssim(rendered_image, gt_image).mean()
```

**Returns**: Tuple of (average_psnr, average_ssim)

---

### log_training_results() (Lines 284-290)

**Purpose**: Save training metric logs to text files  
**Parameters**:
- `model_path`: Output directory
- `filename`: Name of log file (e.g., "psnr_log.txt")
- `log_data`: List of (iteration, value) tuples

**Format**:
```
7000: 28.4523
14000: 30.1234
...
```

---

## Data Flow

### Input Pipeline

**COLMAP Data Loading** (`scene/colmap_loader.py`):

1. **Structure from Motion Output**:
   - `cameras.bin`: Intrinsic parameters (focal length, principal point, distortion)
   - `images.bin`: Extrinsic parameters (rotation, translation), image paths
   - `points3D.bin`: 3D point positions, colors, track information

2. **Image Loading**: RGB images loaded from paths specified in `images.bin`

3. **Optional Data**:
   - Monocular depth maps: `depth/` directory with per-image depth and confidence masks
   - Exposure data: Text file with per-image exposure values

**Initialization** (`scene/gaussian_model.py:create_from_pcd`):

```python
points3D = basic_point_cloud.points  # Nx3 positions
colors = basic_point_cloud.colors    # Nx3 RGB [0,1]

self._xyz = nn.Parameter(torch.tensor(points3D, dtype=torch.float, device="cuda"))
self._features_dc = nn.Parameter(features[:, :1, :].transpose(1, 2).contiguous())
self._scaling = nn.Parameter(torch.log(torch.sqrt(dists)))  # Log-space for stability
self._rotation = nn.Parameter(torch.zeros((points3D.shape[0], 4)))
self._rotation[:, 0] = 1  # Identity quaternion
self._opacity = nn.Parameter(inverse_sigmoid(0.1 * torch.ones(...)))
```

Initial scales set to distance to 3rd nearest neighbor, ensuring reasonable Gaussian sizes.

### Forward Pass (Rendering)

**3D to 2D Projection** (`gaussian_renderer/__init__.py`):

1. **Covariance Construction**:
   ```
   S = diag(exp(scaling))  # 3x3 scale matrix
   R = quaternion_to_matrix(rotation)  # 3x3 rotation
   Σ_world = R @ S @ S^T @ R^T  # 3D covariance in world space
   ```

2. **View Transformation**:
   ```
   Σ_camera = J @ W @ Σ_world @ W^T @ J^T
   ```
   Where:
   - `W`: World-to-camera rotation/translation
   - `J`: Jacobian of perspective projection
   
3. **2D Covariance**:
   ```
   Σ_2d = [Σ_camera[0,0]  Σ_camera[0,1]]
          [Σ_camera[1,0]  Σ_camera[1,1]]
   ```

4. **Tile-based Rasterization**:
   - Sort Gaussians by depth per 16x16 tile
   - Alpha-blend in front-to-back order
   - Accumulate color and transmittance

**Output Tensors**:
- `render`: RGB image [3, H, W]
- `depth`: Depth map [1, H, W] (distance to camera)
- `viewspace_points`: 2D gradients w.r.t. screen-space positions
- `visibility_filter`: Boolean mask of visible Gaussians
- `radii`: Screen-space radii in pixels

### Backward Pass

**Gradient Flow**:

1. **Loss → Rendered Image**: Standard autograd through L1, SSIM, etc.

2. **Rendered Image → 2D Gaussians**: Custom CUDA kernels in rasterizer compute:
   - ∂L/∂color_i for each Gaussian
   - ∂L/∂α_i (opacity gradients)
   - ∂L/∂μ_2d (position gradients in screen space)
   - ∂L/∂Σ_2d (covariance gradients)

3. **2D → 3D Parameters**: Chain rule through projection:
   - ∂L/∂xyz via viewspace_point_tensor
   - ∂L/∂scaling and ∂L/∂rotation via Σ_world
   - ∂L/∂opacity via sigmoid transform
   - ∂L/∂SH_coeffs via spherical harmonic evaluation

**Gradient Accumulation for Densification**:

```python
# scene/gaussian_model.py:add_densification_stats
self.xyz_gradient_accum[visibility_filter] += torch.norm(viewspace_point_tensor.grad[visibility_filter, :2], dim=-1, keepdim=True)
self.denom[visibility_filter] += 1
```

Average gradient magnitude guides cloning/splitting decisions.

### Output Generation

**Model Saves** (`scene/__init__.py:save`):

1. **Point Cloud** (PLY format):
   ```
   x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 f_rest_0 ... opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
   ```
   
2. **Checkpoint** (PyTorch format):
   ```python
   {
       'active_sh_degree': int,
       'xyz': Tensor,
       'features_dc': Tensor,
       'features_rest': Tensor,
       'scaling': Tensor,
       'rotation': Tensor,
       'opacity': Tensor,
       'max_radii2D': Tensor,
       'xyz_gradient_accum': Tensor,
       'denom': Tensor,
       'optimizer_state_dict': dict,
       'spatial_lr_scale': float,
   }
   ```

**Logs**:
- `cfg_args`: Text file with all configuration parameters
- `psnr_log.txt`, `ssim_log.txt`: Per-iteration metrics
- TensorBoard events: Scalars (losses, metrics), images (renders, GT)

---

## Limitations

### 1. Memory Constraints

**Gaussian Count Scaling**: Memory usage grows linearly with number of Gaussians. Dense scenes can require 5-10 million Gaussians (5-10 GB GPU memory for parameters alone). The rasterizer also maintains per-Gaussian metadata (radii, gradients) during training.

**Mitigation**: 
- Use `optimizer_type="sparse_adam"` to reduce memory by only storing gradient statistics for visible Gaussians
- Increase `densify_grad_threshold` to limit total point count
- Reduce `sh_degree` from 3 to 2 or 1 (fewer spherical harmonic bands)

### 2. Training Time

**Iteration Speed**: Rendering is fast (~5-10ms for 800x600 on RTX 3090) but requires 30,000 iterations for convergence. Full training takes 30-60 minutes for small scenes, 2-4 hours for complex ones.

**Bottlenecks**:
- Densification (sorting and cloning operations) at iterations 500-15000
- Test evaluation every 1000 iterations (renders all test views)
- SH degree increases every 1000 iterations (larger color computations)

**Mitigation**: 
- Reduce `test_iterations` frequency
- Use smaller test sets during training
- Skip bundle training for faster convergence on small scenes

### 3. Camera Calibration Sensitivity

**Requirement**: Relies on accurate COLMAP reconstruction. Poor calibration (errors in intrinsics/extrinsics) causes:
- Misaligned Gaussians (floaters)
- Blurry regions (averaging misaligned content)
- Degraded novel view synthesis

**Indicators**:
- High loss but poor visual quality
- Points densifying in free space
- Inconsistent depth maps

**Mitigation**:
- Verify COLMAP output quality before training
- Use bundle adjustment to refine calibration
- Enable depth regularization if monocular depth available

### 4. Densification Hysteresis

**Issue**: The densification process is irreversible within an iteration window. Once split, Gaussians are not automatically merged. Over-densification in early iterations can create redundant points that persist.

**Manifestation**:
- Unnecessarily high point count
- Increased memory and rendering cost
- Minimal quality improvement after iteration 15000

**Mitigation**:
- Tune `densify_grad_threshold` (default 0.0002) - higher values reduce splitting
- Adjust `opacity_reset_interval` to prune more aggressively
- Use the pruning pass after training (not implemented in base code)

### 5. Bundle Training Complexity

**Configuration Burden**: Bundle training introduces hyperparameters:
- `camera_order`: Choice between covisibility, PCA, or clustering
- Group size (hardcoded to 20)
- Interval scheduling (iterations 80-99 per 100)

**When to Use**:
- ✅ Large scenes (>200 images) with spatially correlated cameras
- ✅ Scenes with distinct regions requiring focused densification
- ❌ Small scenes (<50 images) - overhead exceeds benefit
- ❌ Scenes with uniform camera distribution

**Debugging Difficulty**: Harder to diagnose convergence issues due to non-random sampling.

### 6. Depth Regularization Limitations

**Dependency on Monocular Depth Quality**: When `depth_reliable=True`, the system trusts monocular depth predictions. Errors in depth estimation (common in textureless regions, thin structures) can:
- Bias geometry toward incorrect depths
- Create inconsistencies with photometric loss
- Degrade multi-view consistency

**Weight Scheduling**: Exponential decay from `depth_l1_weight_init` (default 2.0) to `depth_l1_weight_final` (default 0.0) is fixed. Scenes with varying depth reliability might benefit from adaptive weighting.

**No Uncertainty Modeling**: Depth mask is binary (reliable/unreliable). A confidence-weighted approach would be more robust.

### 7. Loss Function Limitations

**SSIM Perceptual Gap**: While SSIM is better than L1 for perceptual quality, it still doesn't fully capture human perception. The 80/20 L1/SSIM weighting is empirically chosen but may not be optimal for all scenes.

**No Adversarial Component**: Unlike NeRF methods with GAN components, this approach lacks explicit adversarial training for realism. Fine details and textures may appear smoothed.

**Laplacian Pyramid Overhead**: When `enable_ds_lap=True`, the multi-scale loss adds 20-30% computational cost but often provides marginal improvement over SSIM alone.

### 8. Exposure Compensation Scope

**Per-Camera Only**: Exposure parameters are per-camera, not per-region. Scenes with spatially varying lighting (shadows, indoor/outdoor transitions) cannot be fully modeled.

**Training vs. Inference Mismatch**: When `train_test_exp=True`, exposure is optimized during training but may not generalize to test cameras. This can cause brightness/color shifts in novel views.

### 9. No Semantic Understanding

**Geometric Representation Only**: Gaussians represent shape and appearance but lack semantic labels. This prevents:
- Object-level editing (e.g., "remove the chair")
- Semantic-aware densification (e.g., prioritizing faces)
- Material decomposition (disentangling lighting and reflectance)

### 10. Submodule Dependencies

**External Code**: Relies on compiled CUDA extensions:
- `diff_gaussian_rasterization`: Core rasterizer (from submodules/)
- `fused_ssim`: Optimized SSIM implementation
- `SparseGaussianAdam`: Memory-efficient optimizer

**Installation Complexity**: Building these requires:
- CUDA toolkit matching PyTorch version
- Compatible C++ compiler
- Proper include paths for PyTorch headers

**Portability**: Training is CUDA-only; no CPU or other accelerator support.

---

## Utility Module Reference

### utils/loss_utils.py

**Core Functions**:
- `l1_loss(network_output, gt)`: Mean absolute error
- `ssim(img1, img2, window_size=11)`: Structural similarity index with Gaussian weighting
- `InvDepthSmoothnessLoss`: Edge-aware depth smoothness using image gradients
- `laplacian_pyramid_loss`: Multi-scale texture comparison (4 levels by default)

**Implementation Details**:
- SSIM uses 11x11 Gaussian kernel (σ=1.5)
- Depth smoothness exponentially weights by image gradient magnitude (α=10)
- Laplacian pyramid performs Gaussian blur + downsample iteratively

### utils/bundle_utils.py

**Core Functions**:
- `cluster_cameras(colmap_path, camera_order)`: Orders training cameras by covisibility, PCA, or clustering
- `build_covisibility_matrix(images, points3D)`: Constructs NxN matrix counting shared points
- `create_sequence_from_covisibility_graph(graph)`: Greedy traversal starting from highest-degree node
- `bundle_start_index_generator(sorted_keys, group_size)`: Generates sliding window indices

**Covisibility Algorithm**:
1. For each 3D point, increment matrix[i,j] for all camera pairs (i,j) viewing it
2. Construct graph with edges weighted by covisibility count
3. Start from camera with most connections (>= min_covisibility threshold)
4. Greedily select next camera with highest covisibility to current
5. If stuck, jump to unvisited camera with best connection to sequence

### utils/general_utils.py

**Core Functions**:
- `get_expon_lr_func(lr_init, lr_final, max_steps)`: Returns exponential decay scheduler
- `build_rotation(r)`: Converts quaternion to 3x3 rotation matrix
- `build_scaling_rotation(s, r)`: Computes R @ S for covariance construction
- `safe_state(silent)`: Seeds RNG and redirects stdout with timestamps

**Mathematical Details**:
- Quaternion to rotation: Standard formula avoiding gimbal lock
- Learning rate: `lr = delay_rate * exp(log(lr_init) * (1-t) + log(lr_final) * t)` where t = step/max_steps
- Delay rate: Smooth warmup via reverse cosine if lr_delay_steps > 0

### utils/image_utils.py

**Core Functions**:
- `psnr(img1, img2)`: Peak signal-to-noise ratio (20 * log10(1/√MSE))
- `mse(img1, img2)`: Mean squared error

**Tensor Handling**: Operates on batched tensors [B, C, H, W], reducing over all dimensions except batch.

### utils/aug_utils.py

**Purpose**: Quadtree-based point cloud augmentation (experimental feature not used in base training)

**Core Functions**:
- `quadtree_decomposition(img, threshold, min_pixel_size)`: Recursive subdivision based on variance
- `find_depth_from_nn(image, leaf_nodes, points3d_pix)`: Interpolates depth via nearest 3 neighbors
- `image_quadtree_augmentation(...)`: Generates new 3D points in unobserved image regions

**Algorithm**:
1. Build quadtree where leaves have variance < threshold
2. Identify leaves without projected 3D points
3. For each empty leaf, sample random pixel
4. Find 3 nearest projected 3D points
5. Fit plane through neighbors, compute depth along ray
6. Transform to world coordinates and add to point cloud

**Use Case**: Densifying regions with sparse SfM reconstruction, particularly textureless areas.

### utils/graphics_utils.py

**Core Functions**:
- `getWorld2View2(R, t, translate, scale)`: Constructs 4x4 world-to-camera matrix with normalization
- `getProjectionMatrix(znear, zfar, fovX, fovY)`: OpenGL-style projection matrix
- `fov2focal(fov, pixels)` / `focal2fov(focal, pixels)`: Field-of-view conversions

**Coordinate Systems**:
- World: Right-handed, arbitrary origin from COLMAP
- Camera: OpenCV convention (Z forward, Y down, X right)
- Clip: OpenGL convention (Z out of screen)

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-17  
**Code Version**: Based on train.py (449 lines)
