#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import os
import sys
import uuid
import torch

from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state, get_linear_noise_func

from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)

    # -------------------------------------------------------------------------
    # Explicit per-Gaussian ODE-GS design:
    #
    # GaussianModel now owns both:
    #   1. normal 3DGS parameters:
    #        xyz, rotation, scaling, opacity, SH features
    #
    #   2. explicit ODE motion parameters:
    #        ode_A, ode_b, ode_omega, ode_kappa
    #
    # There is no separate DeformModel anymore.
    # -------------------------------------------------------------------------
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    # This must initialize optimizer groups for BOTH:
    #   - normal Gaussian parameters
    #   - explicit ODE parameters inside GaussianModel
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0

    progress_bar = tqdm(range(opt.iterations), desc="Training progress")

    smooth_term = get_linear_noise_func(
        lr_init=0.1,
        lr_final=1e-15,
        lr_delay_mult=0.01,
        max_steps=20000,
    )

    for iteration in range(1, opt.iterations + 1):
        # ---------------------------------------------------------------------
        # Optional network GUI connection.
        # Kept from original training loop.
        # ---------------------------------------------------------------------
        if network_gui.conn is None:
            network_gui.try_connect()

        while network_gui.conn is not None:
            try:
                net_image_bytes = None

                (
                    custom_cam,
                    do_training,
                    pipe.do_shs_python,
                    pipe.do_cov_python,
                    keep_alive,
                    scaling_modifier,
                ) = network_gui.receive()

                if custom_cam is not None:
                    # GUI preview uses canonical Gaussians without explicit time.
                    # If you want GUI-time rendering later, you can also call
                    # gaussians.compute_ode_deformation(...) here.
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifier,
                    )["render"]

                    net_image_bytes = memoryview(
                        (
                            torch.clamp(net_image, min=0, max=1.0)
                            * 255
                        )
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )

                network_gui.send(net_image_bytes, dataset.source_path)

                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break

            except Exception:
                network_gui.conn = None

        iter_start.record()

        # ---------------------------------------------------------------------
        # Increase SH degree every 1000 iterations.
        # ---------------------------------------------------------------------
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ---------------------------------------------------------------------
        # Pick a random training camera.
        # ---------------------------------------------------------------------
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1.0 / total_frame

        viewpoint_cam = viewpoint_stack.pop(
            randint(0, len(viewpoint_stack) - 1)
        )

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()

        fid = viewpoint_cam.fid

        # ---------------------------------------------------------------------
        # Explicit ODE deformation.
        #
        # Before warm-up:
        #   use canonical Gaussians directly.
        #
        # After warm-up:
        #   ask GaussianModel to compute deformation from its own ODE params.
        # ---------------------------------------------------------------------
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            if dataset.is_blender:
                ast_noise = 0.0
            else:
                ast_noise = (
                    torch.randn(1, 1, device="cuda")
                    .expand(N, -1)
                    * time_interval
                    * smooth_term(iteration)
                )

            d_xyz, d_rotation, d_scaling = gaussians.compute_ode_deformation(
                time_input + ast_noise,
                is_6dof=dataset.is_6dof,
            )

        # ---------------------------------------------------------------------
        # Render.
        # ---------------------------------------------------------------------
        render_pkg_re = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            d_xyz,
            d_rotation,
            d_scaling,
            dataset.is_6dof,
        )

        image = render_pkg_re["render"]
        viewspace_point_tensor = render_pkg_re["viewspace_points"]
        visibility_filter = render_pkg_re["visibility_filter"]
        radii = render_pkg_re["radii"]

        # ---------------------------------------------------------------------
        # Photometric loss.
        # Gradients now flow only into GaussianModel, because ODE parameters
        # are supposed to be stored inside GaussianModel.
        # ---------------------------------------------------------------------
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        )

        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device("cpu")

        with torch.no_grad():
            # -----------------------------------------------------------------
            # Progress bar.
            # -----------------------------------------------------------------
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{ema_loss_for_log:.7f}"}
                )
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            # -----------------------------------------------------------------
            # Track max radii for pruning.
            # -----------------------------------------------------------------
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )

            # -----------------------------------------------------------------
            # Logging / validation.
            # -----------------------------------------------------------------
            cur_psnr = training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                dataset.load2gpu_on_the_fly,
                dataset.is_6dof,
            )

            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            # -----------------------------------------------------------------
            # Save Gaussians.
            #
            # Since ODE params are now part of GaussianModel, scene.save(...)
            # must eventually save them too.
            # -----------------------------------------------------------------
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # -----------------------------------------------------------------
            # Densification.
            #
            # Later, GaussianModel.densify_and_prune(...) must also clone/split/
            # prune the explicit ODE tensors:
            #   ode_A, ode_b, ode_omega, ode_kappa
            # -----------------------------------------------------------------
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re[
                    "viewspace_points_densify"
                ]

                gaussians.add_densification_stats(
                    viewspace_point_tensor_densify,
                    visibility_filter,
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )

                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )
                    print(f"[ITER {iteration}] Num Gaussians = {gaussians.get_xyz.shape[0]}")
                
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background
                    and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # -----------------------------------------------------------------
            # Optimizer step.
            #
            # Only GaussianModel optimizer exists now.
            # The Gaussian optimizer must include both standard Gaussian params
            # and explicit ODE params.
            # -----------------------------------------------------------------
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)

    print(f"Best PSNR = {best_psnr} in Iteration {best_iteration}")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())

        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None

    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss_func,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    load2gpu_on_the_fly,
    is_6dof=False,
):
    if tb_writer:
        tb_writer.add_scalar(
            "train_loss_patches/l1_loss",
            Ll1.item(),
            iteration,
        )
        tb_writer.add_scalar(
            "train_loss_patches/total_loss",
            loss.item(),
            iteration,
        )
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    test_psnr = 0.0

    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        validation_configs = (
            {
                "name": "test",
                "cameras": scene.getTestCameras(),
            },
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")

                for idx, viewpoint in enumerate(config["cameras"]):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()

                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

                    d_xyz, d_rotation, d_scaling = (
                        scene.gaussians.compute_ode_deformation(
                            time_input,
                            is_6dof=is_6dof,
                        )
                    )

                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            scene.gaussians,
                            *renderArgs,
                            d_xyz,
                            d_rotation,
                            d_scaling,
                            is_6dof,
                        )["render"],
                        0.0,
                        1.0,
                    )

                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"),
                        0.0,
                        1.0,
                    )

                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device("cpu")

                    if tb_writer and idx < 5:
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(
                                    viewpoint.image_name
                                ),
                                gt_image[None],
                                global_step=iteration,
                            )

                l1_test = l1_loss_func(images, gts)
                psnr_test = psnr(images, gts).mean()

                if (
                    config["name"] == "test"
                    or len(validation_configs[0]["cameras"]) == 0
                ):
                    test_psnr = psnr_test

                print(
                    f"\n[ITER {iteration}] Evaluating {config['name']}: "
                    f"L1 {l1_test} PSNR {psnr_test}"
                )

                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss",
                        l1_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr",
                        psnr_test,
                        iteration,
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram",
                scene.gaussians.get_opacity,
                iteration,
            )
            tb_writer.add_scalar(
                "total_points",
                scene.gaussians.get_xyz.shape[0],
                iteration,
            )

        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)

    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[5000, 6000, 7000] + list(range(10000, 40001, 1000)),
    )

    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[7000, 10000, 20000, 30000, 40000],
    )

    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
    )

    print("\nTraining complete.")