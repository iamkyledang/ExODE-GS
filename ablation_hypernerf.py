"""
Ablation study for ODE-GS on the HyperNeRF interpolation dataset.

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

Default scene: interp_aleks-teapot (HyperNeRF interpolation dataset).
Outputs land in:
  output_ablation_hypernerf/<variant>/<scene>/

Usage:
  python ablation_hypernerf.py                                    # run all 5 variants on interp_aleks-teapot
  python ablation_hypernerf.py --scenes interp_hand interp_cut-lemon
  python ablation_hypernerf.py --variants full no_A no_omega
  python ablation_hypernerf.py --skip_existing          # skip already-finished combos (default ON)
  python ablation_hypernerf.py --no_skip_existing       # re-run everything
"""

import subprocess
import os
import json
import argparse

# ---------------------------------------------------------------------------
# Scenes  (outer folder name → inner dataset subfolder auto-detected)
# ---------------------------------------------------------------------------
ALL_SCENES = [
    "interp_aleks-teapot",
    "interp_chickchicken",
    "interp_cut-lemon",
    "interp_hand",
    "interp_slice-banana",
    "interp_torchocolate",
]

DEFAULT_SCENE = "interp_aleks-teapot"

ITERATIONS = 40_000

# ---------------------------------------------------------------------------
# Ablation variants
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
    return os.path.join("output_ablation_hypernerf", variant_name, scene)


def is_done(variant_name, scene):
    """Return True if results.json with PSNR already exists for this variant/scene pair."""
    rpath = os.path.join(output_dir(variant_name, scene), "results.json")
    if not os.path.exists(rpath):
        return False
    try:
        with open(rpath) as f:
            data = json.load(f)
        for key, val in data.items():
            if key.startswith("ours_") and isinstance(val, dict) and "PSNR" in val:
                return True
    except Exception:
        pass
    return False


def find_inner_dir(scene):
    """Return the single inner dataset subfolder path for a HyperNeRF scene."""
    outer = os.path.join("..", "hypernerf_interp", scene)
    inner_dirs = [
        d for d in os.listdir(outer)
        if os.path.isdir(os.path.join(outer, d))
    ]
    assert len(inner_dirs) == 1, (
        f"Expected exactly one dataset subfolder in {outer}, got: {inner_dirs}"
    )
    return os.path.join(outer, inner_dirs[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ODE-GS ablation study on HyperNeRF")
    parser.add_argument(
        "--scenes", nargs="+", default=[DEFAULT_SCENE], choices=ALL_SCENES,
        metavar="SCENE",
        help=f"HyperNeRF scenes to evaluate (default: {DEFAULT_SCENE})",
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
    print(f"ODE-GS Ablation Study  —  HyperNeRF Dataset")
    print(f"  Variants : {[v['name'] for v in selected_variants]}")
    print(f"  Scenes   : {selected_scenes}")
    print(f"  Total    : {total} experiment(s)")
    print(f"  Output   : output_ablation_hypernerf/<variant>/<scene>/")
    print(f"{'='*70}\n")

    for variant in selected_variants:
        v_name  = variant["name"]
        v_desc  = variant["desc"]
        v_extra = variant["extra_args"]

        for scene in selected_scenes:
            done += 1
            out    = output_dir(v_name, scene)
            source = find_inner_dir(scene)

            print(f"\n{'='*70}")
            print(f"[{done}/{total}]  Variant: {v_name}  |  Scene: {scene}")
            print(f"  {v_desc}")
            print(f"  Source : {source}")
            print(f"  Output : {out}")
            print(f"{'='*70}")

            if args.skip_existing and is_done(v_name, scene):
                print(f"  results.json with PSNR found — skipping.\n")
                continue

            # ----------------------------------------------------------------
            # 1. Train
            # ----------------------------------------------------------------
            # HyperNeRF interp: no --white_background, no --is_6dof
            run([
                "python", "train.py",
                "-s", source,
                "-m", out,
                "--eval",
                "--load2gpu_on_the_fly",
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
    print("ABLATION SUMMARY  —  HyperNeRF")
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

    print(f"\nOutputs are in:  output_ablation_hypernerf/<variant>/<scene>/")
    print("All done.\n")


if __name__ == "__main__":
    main()
