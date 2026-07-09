"""
Ablation study for ODE-GS on the NeRF-DS dataset.

The five new parameters introduced over vanilla 3DGS are:
  1. ode_A     – (N,3,3) matrix in translation ODE  dp/dtau = A p + b
  2. ode_b     – (N,3)   bias  in translation ODE
  3. ode_omega – (N,3)   angular-velocity in rotation ODE  dQ/dtau = hat(omega) Q
  4. ode_kappa – (N,3)   log-scale rate          d_scale = kappa * tau
  5. warm_up   – training iterations before ODE deformation is applied

Five experiments in total:
  Exp 0 – full     : all four ODE tensors active, warm_up = 3000 (default)
  Exp 1 – no_A     : ode_A frozen (lr = 0); tests contribution of translation matrix
  Exp 2 – no_b     : ode_b frozen (lr = 0); tests contribution of translation bias
  Exp 3 – no_omega : ode_omega frozen (lr = 0); tests contribution of rotation ODE
  Exp 4 – no_kappa : ode_kappa frozen (lr = 0); tests contribution of scale ODE

Default scene: as_novel_view (NeRF-DS dataset).
Outputs land in:
  output_ablation_nerf_ds/<variant>/<scene>/

Usage:
  python ablation_nerf_ds.py                          # run all 5 variants on as_novel_view
  python ablation_nerf_ds.py --scenes cup_novel_view bell_novel_view
  python ablation_nerf_ds.py --variants full no_A no_omega
  python ablation_nerf_ds.py --skip_existing          # skip already-finished combos (default ON)
  python ablation_nerf_ds.py --no_skip_existing       # re-run everything
"""

import subprocess
import os
import json
import argparse

# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------
ALL_SCENES = [
    "as_novel_view",
    "basin_novel_view",
    "bell_novel_view",
    "cup_novel_view",
    "plate_novel_view",
    "press_novel_view",
    "sieve_novel_view",
]

DEFAULT_SCENE = "as_novel_view"

ITERATIONS = 40_000

# ---------------------------------------------------------------------------
# Ablation variants
# Each entry:
#   name       – short identifier used in the output folder name
#   desc       – human-readable description printed to console
#   extra_args – list of additional CLI tokens passed to train.py
# ---------------------------------------------------------------------------
ABLATIONS = [
    {
        "name": "full",
        "desc": "Full ODE — all four ODE tensors (A, b, omega, kappa) active; warm_up=3000",
        "extra_args": [],
    },
    {
        "name": "no_A",
        "desc": "w/o ode_A — translation matrix frozen (lr=0); only ode_b drives translation",
        "extra_args": ["--ode_A_lr", "0"],
    },
    {
        "name": "no_b",
        "desc": "w/o ode_b — translation bias frozen (lr=0); only ode_A drives translation",
        "extra_args": ["--ode_b_lr", "0"],
    },
    {
        "name": "no_omega",
        "desc": "w/o ode_omega — rotation ODE frozen (lr=0); Gaussians do not rotate over time",
        "extra_args": ["--ode_omega_lr", "0"],
    },
    {
        "name": "no_kappa",
        "desc": "w/o ode_kappa — scale ODE frozen (lr=0); Gaussian scales fixed over time",
        "extra_args": ["--ode_kappa_lr", "0"],
    },
]

ABLATION_NAMES = [v["name"] for v in ABLATIONS]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def output_dir(variant_name, scene):
    return os.path.join("output_ablation_nerf_ds", variant_name, scene)


def is_done(variant_name, scene):
    """Return True if results.json already exists for this variant/scene pair."""
    return os.path.exists(os.path.join(output_dir(variant_name, scene), "results.json"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ODE-GS ablation study on NeRF-DS")
    parser.add_argument(
        "--scenes", nargs="+", default=ALL_SCENES, choices=ALL_SCENES,
        metavar="SCENE",
        help="NeRF-DS scenes to evaluate (default: all scenes)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=ABLATION_NAMES, choices=ABLATION_NAMES,
        metavar="VARIANT",
        help="Ablation variants to run (default: all 5)",
    )
    parser.add_argument(
        "--skip_existing", dest="skip_existing", action="store_true", default=True,
        help="Skip scene/variant combos where results.json already exists (default: on)",
    )
    parser.add_argument(
        "--no_skip_existing", dest="skip_existing", action="store_false",
        help="Re-run even if results.json already exists",
    )
    args = parser.parse_args()

    selected_variants = [v for v in ABLATIONS if v["name"] in args.variants]
    selected_scenes   = args.scenes

    total = len(selected_variants) * len(selected_scenes)
    done  = 0

    print(f"\n{'='*70}")
    print(f"ODE-GS Ablation Study  —  NeRF-DS Dataset")
    print(f"  Variants : {[v['name'] for v in selected_variants]}")
    print(f"  Scenes   : {selected_scenes}")
    print(f"  Total    : {total} experiment(s)")
    print(f"  Output   : output_ablation_nerf_ds/<variant>/<scene>/")
    print(f"{'='*70}\n")

    for variant in selected_variants:
        v_name  = variant["name"]
        v_desc  = variant["desc"]
        v_extra = variant["extra_args"]

        for scene in selected_scenes:
            done += 1
            out    = output_dir(v_name, scene)
            source = os.path.join("..", "nerf_ds", scene)

            print(f"\n{'='*70}")
            print(f"[{done}/{total}]  Variant: {v_name}  |  Scene: {scene}")
            print(f"  {v_desc}")
            print(f"  Source : {source}")
            print(f"  Output : {out}")
            print(f"{'='*70}")

            if args.skip_existing and is_done(v_name, scene):
                print(f"  results.json found — skipping.\n")
                continue

            # ----------------------------------------------------------------
            # 1. Train
            # ----------------------------------------------------------------
            # NeRF-DS uses --is_6dof for 6-DoF camera poses
            run([
                "python", "train.py",
                "-s", source,
                "-m", out,
                "--eval",
                "--is_6dof",
            ] + v_extra)

            # ----------------------------------------------------------------
            # 2. Render test set
            # ----------------------------------------------------------------
            run([
                "python", "render.py",
                "-m", out,
                "--skip_train",
                "--mode", "render",
            ])

            # ----------------------------------------------------------------
            # 3. Side-by-side comparison video (render | gt)
            # ----------------------------------------------------------------
            renders_dir = os.path.join(out, "test", f"ours_{ITERATIONS}", "renders", "%05d.png")
            gt_dir      = os.path.join(out, "test", f"ours_{ITERATIONS}", "gt",      "%05d.png")
            out_video   = os.path.join(out, "test", f"ours_{ITERATIONS}", "comparison.mp4")
            run([
                "ffmpeg", "-y",
                "-framerate", "30", "-i", renders_dir,
                "-framerate", "30", "-i", gt_dir,
                "-filter_complex", "hstack,pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                out_video,
            ])

            # ----------------------------------------------------------------
            # 4. Metrics
            # ----------------------------------------------------------------
            run(["python", "metrics.py", "-m", out])

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("ABLATION SUMMARY  —  NeRF-DS")
    print(f"{'='*70}")

    col_w = 14
    header = f"{'Scene':<26}" + "".join(f"{'PSNR ' + v['name']:>{col_w}}" for v in selected_variants)
    print(header)
    print("-" * len(header))

    for scene in selected_scenes:
        row = f"{scene:<26}"
        for variant in selected_variants:
            out          = output_dir(variant["name"], scene)
            results_path = os.path.join(out, "results.json")
            psnr_str     = "N/A"
            if os.path.exists(results_path):
                try:
                    with open(results_path) as f:
                        data = json.load(f)
                    for key, val in data.items():
                        if key.startswith("ours_") and isinstance(val, dict) and "PSNR" in val:
                            psnr_str = f"{val['PSNR']:.4f}"
                            break
                except Exception:
                    psnr_str = "ERR"
            row += f"{psnr_str:>{col_w}}"
        print(row)

    print(f"\nOutputs are in:  output_ablation_nerf_ds/<variant>/<scene>/")
    print("All done.\n")


if __name__ == "__main__":
    main()
