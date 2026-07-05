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

from pathlib import Path
import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

# def readImages(renders_dir, gt_dir):
#     """Return sorted filenames only; images are loaded one-at-a-time during evaluation."""
#     image_names = sorted(os.listdir(renders_dir))
#     return renders_dir, gt_dir, image_names
#
#
# def _load_image(path):
#     return tf.to_tensor(Image.open(path)).unsqueeze(0)[:, :3, :, :].cuda()


@torch.no_grad()
def compute_flow_epe(renders, gts, image_names, raft_model):
    """Compute mean Flow EPE between consecutive rendered and GT frame pairs."""
    order = sorted(range(len(image_names)), key=lambda i: image_names[i])
    renders = [renders[i] for i in order]
    gts = [gts[i] for i in order]
    epes = []
    for i in range(len(renders) - 1):
        r1, r2 = renders[i], renders[i + 1]
        g1, g2 = gts[i], gts[i + 1]
        def prep(img):
            img = img * 2.0 - 1.0
            _, _, h, w = img.shape
            ph = (8 - h % 8) % 8
            pw = (8 - w % 8) % 8
            return F.pad(img, (0, pw, 0, ph), mode="replicate"), h, w
        r1p, h, w = prep(r1); r2p, _, _ = prep(r2)
        g1p, _, _ = prep(g1); g2p, _, _ = prep(g2)
        flow_r = raft_model(r1p, r2p)[-1][:, :, :h, :w]
        flow_g = raft_model(g1p, g2p)[-1][:, :, :h, :w]
        epes.append(torch.norm(flow_r - flow_g, dim=1).mean().item())
    return float(torch.tensor(epes).mean().item()) if epes else None

# @torch.no_grad()
# def compute_flow_epe_streaming(renders_dir, gt_dir, image_names, raft_model):
#     """Compute mean Flow EPE between consecutive rendered and GT frame pairs (streaming)."""
#     names = sorted(image_names)
#
#     def prep(img):
#         img = img * 2.0 - 1.0
#         _, _, h, w = img.shape
#         ph = (8 - h % 8) % 8
#         pw = (8 - w % 8) % 8
#         return F.pad(img, (0, pw, 0, ph), mode="replicate"), h, w
#
#     epes = []
#     prev_r = prev_g = None
#     for fname in names:
#         r = _load_image(renders_dir / fname)
#         g = _load_image(gt_dir / fname)
#         if prev_r is not None:
#             r1p, h, w = prep(prev_r); r2p, _, _ = prep(r)
#             g1p, _, _ = prep(prev_g); g2p, _, _ = prep(g)
#             flow_r = raft_model(r1p, r2p)[-1][:, :, :h, :w]
#             flow_g = raft_model(g1p, g2p)[-1][:, :, :h, :w]
#             epes.append(torch.norm(flow_r - flow_g, dim=1).mean().item())
#             del r1p, r2p, g1p, g2p, flow_r, flow_g
#             torch.cuda.empty_cache()
#         prev_r = r
#         prev_g = g
#     return float(torch.tensor(epes).mean().item()) if epes else None


def evaluate(model_paths, raft_model=None):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            # Load any stats already written by train.py / render.py into results.json
            _results_path = scene_dir + "/results.json"
            _existing_results = {}
            if os.path.exists(_results_path):
                with open(_results_path) as _f:
                    _existing_results = json.load(_f)
            for _k in ("training_time_seconds", "avg_num_gaussians", "avg_gpu_usage_gb"):
                if _k in _existing_results:
                    full_dict[scene_dir][_k] = _existing_results[_k]

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                if not method.startswith("ours"):
                    continue
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"

                # Pick up fps/num_gaussians written by render.py into results.json
                _render_stats = _existing_results.get(method, {})

                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # renders_dir_p, gt_dir_p, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                # for fname in tqdm(image_names, desc="Metric evaluation progress"):
                #     r = _load_image(renders_dir_p / fname)
                #     g = _load_image(gt_dir_p / fname)
                #     ssims.append(ssim(r, g).item())
                #     psnrs.append(psnr(r, g).item())
                #     lpipss.append(lpips_fn(r, g).item())
                #     del r, g
                #     torch.cuda.empty_cache()

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

                flow_epe = None
                if raft_model is not None:
                    flow_epe = compute_flow_epe(renders, gts, image_names, raft_model)
                    # flow_epe = compute_flow_epe_streaming(renders_dir_p, gt_dir_p, image_names, raft_model)
                    print("  Flow EPE: {:>12.7f}".format(flow_epe) if flow_epe is not None else "  Flow EPE: N/A")

                if "fps" in _render_stats:
                    print("  FPS  : {:>12.5f}".format(_render_stats["fps"]))
                if "num_gaussians" in _render_stats:
                    print("  Num. Gaussians: {:,}".format(_render_stats["num_gaussians"]))
                print("")

                metrics_dict = {"SSIM": torch.tensor(ssims).mean().item(),
                                "PSNR": torch.tensor(psnrs).mean().item(),
                                "LPIPS": torch.tensor(lpipss).mean().item()}
                if flow_epe is not None:
                    metrics_dict["flow_EPE"] = flow_epe
                if "fps" in _render_stats:
                    metrics_dict["fps"] = _render_stats["fps"]
                if "num_gaussians" in _render_stats:
                    metrics_dict["num_gaussians"] = _render_stats["num_gaussians"]
                full_dict[scene_dir][method].update(metrics_dict)
                per_view_dict[scene_dir][method].update(
                    {"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                     "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                     "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

                # per_view_dict[scene_dir][method].update(
                #     {"SSIM": {name: s for s, name in zip(ssims, image_names)},
                #      "PSNR": {name: p for p, name in zip(psnrs, image_names)},
                #      "LPIPS": {name: lp for lp, name in zip(lpipss, image_names)}})

                del renders, gts, ssims, psnrs, lpipss
                # torch.cuda.empty_cache()

            if "training_time_seconds" in full_dict[scene_dir]:
                _t = full_dict[scene_dir]["training_time_seconds"]
                _h = int(_t // 3600); _m = int((_t % 3600) // 60); _s = int(_t % 60)
                print("  Train Time: {:02d}h {:02d}m {:02d}s".format(_h, _m, _s))
            if "avg_num_gaussians" in full_dict[scene_dir]:
                print("  Avg Gaussians (train): {:,}".format(full_dict[scene_dir]["avg_num_gaussians"]))
            if "avg_gpu_usage_gb" in full_dict[scene_dir]:
                print("  Avg GPU (GB): {:>12.4f}".format(full_dict[scene_dir]["avg_gpu_usage_gb"]))

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Unable to compute metrics for model", scene_dir)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--no_flow_epe', action='store_true', help='Disable Flow EPE metric')
    args = parser.parse_args()

    raft_model = None
    if not args.no_flow_epe:
        raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

    evaluate(args.model_paths, raft_model=raft_model)

