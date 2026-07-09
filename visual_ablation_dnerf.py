"""
visual_ablation_dnerf.py
========================
Generate Gaussian trajectory plots and motion heatmaps for every
(variant, scene) pair in output_ablation_dnerf/.

For each pair the script produces two PNG files:
  ablation_dnerf_visual/<variant>_<scene>_trajectories.png
  ablation_dnerf_visual/<variant>_<scene>_heatmap.png

Usage
-----
  python visual_ablation_dnerf.py                          # all variants / all scenes
  python visual_ablation_dnerf.py --variants full no_A     # subset of variants
  python visual_ablation_dnerf.py --scenes lego mutant     # subset of scenes
  python visual_ablation_dnerf.py --top_k 500 --t_steps 30
  python visual_ablation_dnerf.py --skip_existing          # skip already done (default)
  python visual_ablation_dnerf.py --no_skip_existing       # redo everything
"""

import os
import json
import argparse

import numpy as np
from plyfile import PlyData

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection, LineCollection

from PIL import Image
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_ROOT  = "output_ablation_dnerf"
OUTPUT_ROOT = "ablation_dnerf_visual"
ITERATION   = 40_000

ALL_VARIANTS = ["full", "no_A", "no_b", "no_omega", "no_kappa"]
ALL_SCENES   = [
    "bouncingballs",
    "hellwarrior",
    "hook",
    "jumpingjacks",
    "lego",
    "mutant",
    "standup",
    "trex",
]

# Trajectory defaults
DEFAULT_TOP_K   = 300
DEFAULT_T_STEPS = 20

# Heatmap defaults
HEATMAP_SIGMA   = 3.0
HEAT_ALPHA      = 0.7

# Figure resolution
DPI = 150

# ---------------------------------------------------------------------------
# Closed-form ODE helpers (pure numpy)
# ---------------------------------------------------------------------------
def _skew_np(v):
    """v: (N,3) -> (N,3,3) skew-symmetric matrices."""
    N = v.shape[0]
    K = np.zeros((N, 3, 3), dtype=v.dtype)
    K[:, 0, 1] = -v[:, 2];  K[:, 0, 2] =  v[:, 1]
    K[:, 1, 0] =  v[:, 2];  K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1];  K[:, 2, 1] =  v[:, 0]
    return K


def rotation_from_omega(omega, tau):
    """Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²"""
    N = omega.shape[0]
    tau = np.full(N, float(tau)) if np.isscalar(tau) else np.asarray(tau).ravel()
    phi   = omega * tau[:, None]
    theta = np.linalg.norm(phi, axis=-1, keepdims=True)
    axis  = phi / (theta + 1e-8)
    K  = _skew_np(axis)
    K2 = np.einsum("nij,njk->nik", K, K)
    I  = np.eye(3)[None].repeat(N, axis=0).astype(omega.dtype)
    s  = np.sin(theta)[:, :, None]
    c  = (1.0 - np.cos(theta))[:, :, None]
    return I + s * K + c * K2


def translation_from_ode(A, b, tau):
    """
    Augmented-matrix exponential for dp/dτ = Ap + b, p(0)=0.
    A: (N,3,3), b: (N,3), tau: scalar -> (N,3)
    Only called on the small top-K selection; norm-clamped for safety.
    """
    from scipy.linalg import expm

    N = A.shape[0]
    M = np.zeros((N, 4, 4), dtype=np.float64)
    M[:, :3, :3] = A.astype(np.float64)
    M[:, :3,  3] = b.astype(np.float64)
    M_tau = M * float(tau)

    MAX_NORM = 20.0
    norms = np.linalg.norm(M_tau.reshape(N, -1), axis=1)
    scale = np.where(norms > MAX_NORM, MAX_NORM / (norms + 1e-8), 1.0)
    M_tau = M_tau * scale[:, None, None]
    M_tau = np.nan_to_num(M_tau, nan=0.0, posinf=0.0, neginf=0.0)

    p = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        p[i] = expm(M_tau[i])[:3, 3]
    return p


def compute_total_displacement(xyz, ode_A, ode_b):
    """
    Fast norm-based proxy for motion magnitude (no expm over full N).
    Used only for ranking; not for actual trajectory positions.
    """
    b_norm = np.linalg.norm(ode_b, axis=1)
    A_norm = np.linalg.norm(ode_A.reshape(-1, 9), axis=1)
    return b_norm + 0.3 * A_norm


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------
def load_ply(ply_path):
    plydata = PlyData.read(ply_path)
    v = plydata.elements[0]

    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])],
                   axis=1).astype(np.float32)
    N = xyz.shape[0]

    def load_vec(prefix, k):
        return np.stack([np.asarray(v[f"{prefix}_{i}"]) for i in range(k)],
                        axis=1).astype(np.float32)

    ode_A  = load_vec("ode_A", 9).reshape(N, 3, 3)
    ode_b  = load_vec("ode_b", 3)
    ode_omega = load_vec("ode_omega", 3)
    ode_kappa = load_vec("ode_kappa", 3)
    opacity   = np.asarray(v["opacity"]).astype(np.float32)
    scaling   = load_vec("scale", 3)
    rotation  = load_vec("rot", 4)

    return dict(xyz=xyz, ode_A=ode_A, ode_b=ode_b,
                ode_omega=ode_omega, ode_kappa=ode_kappa,
                opacity=opacity, scaling=scaling, rotation=rotation)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def load_cameras(cameras_path):
    with open(cameras_path) as f:
        return json.load(f)


def get_reference_camera(cameras):
    return cameras[0]


def project_points(pts_world, cam):
    pos   = np.array(cam["position"], dtype=np.float64)
    R_cw  = np.array(cam["rotation"], dtype=np.float64)
    R_w2c = R_cw.T
    pts_local = (pts_world - pos[None]) @ R_w2c.T

    fx, fy = cam["fx"], cam["fy"]
    W,  H  = cam["width"], cam["height"]
    z = pts_local[:, 2]
    valid = z > 0.01
    u = np.where(valid, fx * pts_local[:, 0] / (z + 1e-8) + W * 0.5, -9999.0)
    v = np.where(valid, fy * pts_local[:, 1] / (z + 1e-8) + H * 0.5, -9999.0)
    return np.stack([u, v], axis=1), valid


# ---------------------------------------------------------------------------
# Reference background image
# ---------------------------------------------------------------------------
def load_reference_image(model_dir, iteration=ITERATION):
    gt_dir = os.path.join(model_dir, "test", f"ours_{iteration}", "gt")
    if os.path.isdir(gt_dir):
        frames = sorted(os.listdir(gt_dir))
        if frames:
            return np.array(Image.open(os.path.join(gt_dir, frames[0])).convert("RGB"))
    return None


# ---------------------------------------------------------------------------
# Trajectory visualisation
# ---------------------------------------------------------------------------
def make_trajectory_figure(data, cam, ref_img, variant, scene, top_k, t_steps):
    xyz         = data["xyz"]
    ode_A       = data["ode_A"]
    ode_b       = data["ode_b"]
    scaling     = data["scaling"]
    opacity_raw = data["opacity"]

    motion  = compute_total_displacement(xyz, ode_A, ode_b)
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw.ravel()))
    score   = motion * opacity
    top_idx = np.argsort(score)[::-1][:top_k]

    xyz_sel   = xyz[top_idx]
    A_sel     = ode_A[top_idx]
    b_sel     = ode_b[top_idx]
    scale_sel = np.exp(scaling[top_idx])

    taus = np.linspace(0.0, 1.0, t_steps)
    cmap = cm.plasma

    W, H = cam["width"], cam["height"]
    fx   = cam["fx"]

    fig_w = W / DPI * 2.2
    fig_h = H / DPI * 1.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    ax.set_aspect("equal")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis("off")

    if ref_img is not None:
        bg_dark = ref_img.astype(np.float32) / 255.0 * 0.35
        ax.imshow(bg_dark, extent=[0, W, H, 0], aspect="auto", zorder=0)

    for g in range(len(top_idx)):
        pts_w = []
        for tau in taus:
            delta = translation_from_ode(A_sel[g:g+1], b_sel[g:g+1], tau)[0]
            pts_w.append(xyz_sel[g] + delta)
        pts_w = np.array(pts_w)
        proj, valid = project_points(pts_w, cam)
        if valid.sum() < 2:
            continue
        segments, colors = [], []
        for t in range(len(taus) - 1):
            if valid[t] and valid[t+1]:
                segments.append([proj[t], proj[t+1]])
                colors.append(cmap(taus[t]))
        if not segments:
            continue
        ax.add_collection(
            LineCollection(segments, colors=colors, linewidths=0.8, alpha=0.65, zorder=2)
        )

    blob_patches, blob_colors = [], []
    for g in range(len(top_idx)):
        delta  = translation_from_ode(A_sel[g:g+1], b_sel[g:g+1], tau=1.0)[0]
        pos_w  = xyz_sel[g] + delta
        proj1, valid1 = project_points(pos_w[None], cam)
        if not valid1[0]:
            continue
        sc = scale_sel[g]
        z_approx = max(
            float(np.dot(pos_w - np.array(cam["position"]),
                         np.array(cam["rotation"])[2])), 0.5)
        pix_r = np.clip(float(np.mean(sc[[0, 1]])) * fx / z_approx, 3, 60)
        blob_patches.append(Ellipse(xy=(proj1[0, 0], proj1[0, 1]),
                                    width=pix_r * 2, height=pix_r * 1.4, angle=0))
        blob_colors.append(cmap(1.0))

    if blob_patches:
        ax.add_collection(
            PatchCollection(blob_patches, facecolors=[c[:3] for c in blob_colors],
                            alpha=0.35, edgecolors="none", zorder=3)
        )

    sm = cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, aspect=30)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["t=0", "t=0.5", "t=1"], fontsize=6)
    cbar.set_label("Time (start → end)", rotation=90, labelpad=4, fontsize=6)

    ax.set_title(
        f"Gaussian Blob Trajectories – ODE-GS ablation: {variant} / {scene}\n"
        f"top {top_k} moving blobs",
        fontsize=7, fontweight="bold", pad=3,
    )
    fig.tight_layout(pad=0.4)
    return fig


# ---------------------------------------------------------------------------
# Motion heatmap
# ---------------------------------------------------------------------------
def make_heatmap_figure(data, cam, ref_img, variant, scene):
    xyz   = data["xyz"]
    ode_A = data["ode_A"]
    ode_b = data["ode_b"]

    W, H = cam["width"], cam["height"]

    motion   = compute_total_displacement(xyz, ode_A, ode_b)
    thresh   = np.percentile(motion, 50)
    mask     = motion > thresh
    xyz_m    = xyz[mask]
    motion_m = motion[mask]

    proj, valid = project_points(xyz_m, cam)

    heat = np.zeros((H, W), dtype=np.float32)
    if valid.any():
        pu = np.clip(proj[valid, 0].astype(int), 0, W - 1)
        pv = np.clip(proj[valid, 1].astype(int), 0, H - 1)
        np.maximum.at(heat, (pv, pu), motion_m[valid])

    heat_smooth = gaussian_filter(heat, sigma=HEATMAP_SIGMA)
    vmax = heat_smooth.max()
    heat_norm = heat_smooth / vmax if vmax > 0 else heat_smooth

    fig_w = W / DPI * 2.2
    fig_h = H / DPI * 1.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    ax.set_aspect("equal")
    ax.axis("off")

    if ref_img is not None:
        ax.imshow(ref_img, extent=[0, W, H, 0], aspect="auto", zorder=0)
    else:
        ax.imshow(np.full((H, W, 3), 200, dtype=np.uint8),
                  extent=[0, W, H, 0], aspect="auto", zorder=0)

    cmap_heat = cm.inferno
    rgba_heat = cmap_heat(heat_norm)
    rgba_heat[..., 3] = heat_norm * HEAT_ALPHA
    ax.imshow(rgba_heat, extent=[0, W, H, 0], aspect="auto",
              interpolation="bilinear", zorder=2)

    sm = cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap_heat)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, aspect=30)
    cbar.set_label("Normalised motion magnitude", rotation=90, labelpad=4, fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title(
        f"Motion Heatmap – ODE-GS ablation: {variant} / {scene}",
        fontsize=7, fontweight="bold", pad=3,
    )
    fig.tight_layout(pad=0.4)
    return fig


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ODE-GS ablation visualisation – D-NeRF")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                        choices=ALL_VARIANTS, metavar="VARIANT",
                        help="Ablation variants to visualise (default: all)")
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES,
                        choices=ALL_SCENES, metavar="SCENE",
                        help="D-NeRF scenes to visualise (default: all)")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help="Number of highest-motion Gaussians for trajectory plot")
    parser.add_argument("--t_steps", type=int, default=DEFAULT_T_STEPS,
                        help="Number of time steps along each trajectory")
    parser.add_argument("--iteration", type=int, default=ITERATION,
                        help="Checkpoint iteration to load (default: 40000)")
    parser.add_argument("--skip_existing", dest="skip_existing",
                        action="store_true", default=True,
                        help="Skip pairs where output already exists (default: on)")
    parser.add_argument("--no_skip_existing", dest="skip_existing",
                        action="store_false",
                        help="Re-generate even if output already exists")
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    total = len(args.variants) * len(args.scenes)
    done  = 0

    print(f"\n{'='*70}")
    print(f"ODE-GS Ablation Visualisation  —  D-NeRF")
    print(f"  Variants : {args.variants}")
    print(f"  Scenes   : {args.scenes}")
    print(f"  Top-K    : {args.top_k}   T-steps: {args.t_steps}")
    print(f"  Output   : {OUTPUT_ROOT}/")
    print(f"  Total    : {total} pair(s)")
    print(f"{'='*70}\n")

    for variant in args.variants:
        for scene in args.scenes:
            done += 1

            model_dir = os.path.join(INPUT_ROOT, variant, scene)
            ply_path  = os.path.join(model_dir, "point_cloud",
                                     f"iteration_{args.iteration}", "point_cloud.ply")
            cam_path  = os.path.join(model_dir, "cameras.json")

            stem     = f"{variant}_{scene}"
            out_traj = os.path.join(OUTPUT_ROOT, f"{stem}_trajectories.png")
            out_heat = os.path.join(OUTPUT_ROOT, f"{stem}_heatmap.png")

            print(f"\n[{done}/{total}]  {variant} / {scene}")

            files_exist = os.path.exists(out_traj) and os.path.exists(out_heat)
            if args.skip_existing and files_exist:
                print("  All outputs exist — skipping.")
                continue

            if not os.path.exists(ply_path):
                print(f"  [WARN] PLY not found: {ply_path} — skipping.")
                continue
            if not os.path.exists(cam_path):
                print(f"  [WARN] cameras.json not found: {cam_path} — skipping.")
                continue

            print(f"  Loading PLY …")
            data = load_ply(ply_path)
            N    = data["xyz"].shape[0]
            print(f"  N = {N:,} Gaussians")

            cameras = load_cameras(cam_path)
            cam     = get_reference_camera(cameras)

            ref_img = load_reference_image(model_dir, iteration=args.iteration)
            if ref_img is None:
                print("  [INFO] No reference image found; using grey background.")

            if args.skip_existing and os.path.exists(out_traj):
                print(f"  Trajectory image exists — skipping.")
            else:
                print(f"  Rendering trajectories …")
                fig_t = make_trajectory_figure(
                    data, cam, ref_img, variant, scene, args.top_k, args.t_steps)
                fig_t.savefig(out_traj, bbox_inches="tight", dpi=DPI)
                plt.close(fig_t)
                print(f"  Saved: {out_traj}")

            if args.skip_existing and os.path.exists(out_heat):
                print(f"  Heatmap image exists — skipping.")
            else:
                print(f"  Rendering heatmap …")
                fig_h = make_heatmap_figure(data, cam, ref_img, variant, scene)
                fig_h.savefig(out_heat, bbox_inches="tight", dpi=DPI)
                plt.close(fig_h)
                print(f"  Saved: {out_heat}")

    print(f"\n{'='*70}")
    print(f"Done. All outputs in: {OUTPUT_ROOT}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
