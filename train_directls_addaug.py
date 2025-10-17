#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm
import torchvision.transforms.functional as TF

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, InvDepthSmoothnessLoss, laplacian_pyramid_loss
from quadtree_based_initialization import QuadtreeInitDataset, GaussianInitializer
from utils.scheduler_utils import ImageClustering, PartialGroupScheduler

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, 
             opt, 
             pipe, 
             testing_iterations, 
             saving_iterations, 
             checkpoint_iterations, 
             checkpoint, 
             debug_from,
             bundle_training,
             enable_ds_lap,
             lambda_ds,
             lambda_lap,
             n_clusters,
             inv_affinity_matrix,
             similarity_grouping,
             augmentation=False
             ):
    """
    Main training function.

    Args:
        dataset (ModelParams): Dataset parameters.
        opt (OptimizationParams): Optimization parameters.
        pipe (PipelineParams): Pipeline parameters.
        testing_iterations (list): List of iterations to run testing.
        saving_iterations (list): List of iterations to save the model.
        checkpoint_iterations (list): List of iterations to save checkpoints.
        checkpoint (str): Path to a checkpoint to resume training from.
        debug_from (int): Iteration to start debugging from.
        bundle_training (bool): Whether to use bundle training.
        enable_ds_lap (bool): Whether to enable depth smoothness and Laplacian loss.
        lambda_ds (float): Lambda for depth smoothness loss.
        lambda_lap (float): Lambda for Laplacian loss.
        n_clusters (int): Number of clusters for bundle training.
        inv_affinity_matrix (bool): Whether to use inverse affinity matrix for clustering.
        similarity_grouping (bool): Whether to use similarity grouping for bundle training.
        augmentation (bool): Whether to use quadtree-based augmentation.
    """
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Sparse Adam optimizer is not available. Please install the required package.")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Initialize Gaussians using Quadtree-based method
    init_dataset = QuadtreeInitDataset(
        os.path.join(dataset.source_path, dataset.images),
        os.path.join(dataset.source_path, 'sparse/0'),
    )
    # initializer = GaussianInitializer(
    #     gaussians=gaussians,
    #     cameras=scene.getTrainCameras().copy(),
    #     dataset=init_dataset,
    #     pipe=pipe,
    #     background=torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda"),
    #     gs_dataset=dataset,
    #     opt=opt,
    #     args=vars(dataset)
    # )
    # print("Initializing Gaussians with Quadtree-based initialization...")
    # initializer.run()
    # print("Initialization for SfM 3D points complete.")
    # print("This program only initializes colmap SfM 3D points, skips augmentation.")
    init_dataset.directApplicationCov3Ds(GaussianModel=gaussians)
    print("directLS finished")
    import pdb, traceback

    def info(type, value, tb):
        traceback.print_exception(type, value, tb)
        print("\nException occurred! Launching debugger...\n")
        pdb.post_mortem(tb)

    sys.excepthook = info
    gaussians.create_from_augmentor(init_dataset.augmentor)
    # if augmentation:
    #     print("Starting augmentation process...")
    #     initializer.augmentation_mode()
    #     gaussians.create_from_augmentor(initializer.dataset.augmentor)
    #     gaussians.training_setup(opt)
    #     initializer.run_for_augmentation()
        
    #     test_exclude_indices = initializer.dataset.test_only_3d_indices
    #     test_exclude_mask = torch.zeros((gaussians.get_xyz.shape[0]), dtype=torch.bool, device="cuda")
    #     test_exclude_mask[test_exclude_indices] = True
    #     gaussians.prune_points(test_exclude_mask)
    #     print("Augmentation process complete.")
    

    scene.save(0)
    print("Initial Gaussians saved to model path.")
    # exit(0)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    cameras = scene.getTrainCameras().copy()
    if bundle_training:
        clustering = ImageClustering(os.path.join(dataset.source_path, "sparse/0"), n_clusters=n_clusters, inv_affinity_matrix=inv_affinity_matrix)
        scheduler = PartialGroupScheduler(cameras, clustering.ordered_cluster_names,
                                          densify_until_iter=opt.densify_until_iter,
                                          densify_from_iter=opt.densify_from_iter,
                                          debug=False,
                                          similarity_grouping=similarity_grouping,
                                          clustering=clustering)

    psnr_log = []
    ssim_log = []

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifier, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (iteration < opt.iterations or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if bundle_training:
            vind = scheduler.scheduled_training_index(iteration)
            viewpoint_cam = cameras[vind]
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            image *= viewpoint_cam.alpha_mask.cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)) if FUSED_SSIM_AVAILABLE else ssim(image, gt_image)

        ds_loss = InvDepthSmoothnessLoss()(render_pkg["depth"], image) if enable_ds_lap else 0.0
        lap_loss = laplacian_pyramid_loss(image.unsqueeze(0), gt_image.unsqueeze(0)) if enable_ds_lap else 0.0
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + lambda_ds * ds_loss + lambda_lap * lap_loss

        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
        else:
            Ll1depth = torch.tensor(0.0)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth.item() + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "Depth Loss": f"{ema_Ll1depth_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % 1000 == 0:
                # Debug the quantization effect for the first two testing iterations
                debug_quant = iteration in testing_iterations[:2]
                psnr_test, ssim_test = evaluate_test_images(scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp))
                psnr_log.append((iteration, psnr_test))
                ssim_log.append((iteration, ssim_test))
                print(f"\n[ITER {iteration}] Test PSNR: {psnr_test:.4f}, Test SSIM: {ssim_test:.4f}")

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), 1.0 - ssim_value, ds_loss, lap_loss, lambda_ds, lambda_lap, enable_ds_lap)

            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Model")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if bundle_training:
                    if scheduler.densify_and_prune_flag:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                        scheduler.densify_and_prune_flag = False
                    if scheduler.reset_opacity_flag and iteration > 0:
                        gaussians.reset_opacity()
                        scheduler.reset_opacity_flag = False
                else:
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    if (iteration > 0 and iteration % opt.opacity_reset_interval == 0) or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))

    log_training_results(scene.model_path, "psnr_log.txt", psnr_log)
    log_training_results(scene.model_path, "ssim_log.txt", ssim_log)
    print(f"Training complete. Gaussians saved to {scene.model_path}")

def log_training_results(model_path, filename, log_data):
    """Saves training metrics to a log file."""
    log_path = os.path.join(model_path, filename)
    with open(log_path, 'w') as f:
        for iter_num, value in log_data:
            f.write(f"{iter_num}: {value:.4f}\n")
    print(f"Log saved to {log_path}")

def prepare_output_and_logger(args):
    """
    Sets up the output directory and TensorBoard logger.
    """
    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None
    if not TENSORBOARD_FOUND:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss_func, elapsed, testing_iterations, scene, render_func, render_args, ssim_loss, ds_loss, lap_loss, lambda_ds, lambda_lap, enable_ds_lap):
    """
    Logs training progress to TensorBoard and console.
    """
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', ssim_loss.item(), iteration)
        if enable_ds_lap:
            tb_writer.add_scalar('train_loss_patches/ds_loss', ds_loss.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/lap_loss', lap_loss.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/lambda_ds', lambda_ds, iteration)
            tb_writer.add_scalar('train_loss_patches/lambda_lap', lambda_lap, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras']:
                l1_test, psnr_test_val = 0.0, 0.0
                for viewpoint in config['cameras']:
                    image = torch.clamp(render_func(viewpoint, scene.gaussians, *render_args)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer:
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
                    
                    l1_test += l1_loss_func(image, gt_image).mean().double()
                    psnr_test_val += psnr(image, gt_image).mean().double()
                
                psnr_test_val /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.4f} PSNR {psnr_test_val:.4f}")
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint-l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint-psnr", psnr_test_val, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def evaluate_test_images(scene, render_func, render_args):
    """
    Renders all test images and calculates average PSNR and SSIM.
    Includes a debug mode to check the effect of uint8 quantization on PSNR.
    """
    test_cameras = scene.getTestCameras()
    psnr_test = 0.0
    ssim_test = 0.0
    for camera in test_cameras:
        rendered_image = torch.clamp(render_func(camera, scene.gaussians, *render_args)["render"], 0.0, 1.0)
        gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
        psnr_test += psnr(rendered_image, gt_image).mean().double()
        ssim_test += ssim(rendered_image, gt_image).mean().double()
    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    torch.cuda.empty_cache()
    return psnr_test, ssim_test

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--bundle_training", action='store_true', default=False)
    parser.add_argument("--enable_ds_lap", action='store_true', default=False)
    parser.add_argument("--lambda_ds", type=float, default=0.0)
    parser.add_argument("--lambda_lap", type=float, default=0.0)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--inv_affinity_matrix", action='store_true', default=False)
    parser.add_argument("--similarity_grouping", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print(f"Optimizing {args.model_path}")
    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training_args = {
        'dataset': lp.extract(args),
        'opt': op.extract(args),
        'pipe': pp.extract(args),
        'testing_iterations': args.test_iterations,
        'saving_iterations': args.save_iterations,
        'checkpoint_iterations': args.checkpoint_iterations,
        'checkpoint': args.start_checkpoint,
        'debug_from': args.debug_from,
        'bundle_training': args.bundle_training,
        'enable_ds_lap': args.enable_ds_lap,
        'lambda_ds': args.lambda_ds,
        'lambda_lap': args.lambda_lap,
        'n_clusters': args.n_clusters,
        'inv_affinity_matrix': args.inv_affinity_matrix,
        'similarity_grouping': args.similarity_grouping
    }
    
    training(**training_args)

    log_file = os.path.join(args.model_path, "training_args.log")
    with open(log_file, "w") as f:
        f.write("Training Arguments:\n")
        f.write("-" * 50 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
            
    print("\nTraining complete.")
