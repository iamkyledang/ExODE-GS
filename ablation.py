"""
Ablation study for ODE-GS: five new per-Gaussian ODE parameters.

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

Each experiment is run on every scene; outputs land in
  output_ablation/<variant>/<scene>/

Usage:
  python ablation.py                     # run all experiments × all scenes
  python ablation.py --scenes as_novel_view cup_novel_view
  python ablation.py --variants full no_A no_omega
  python ablation.py --skip_existing     # skip already-finished scene/variant combos (default ON)
  python ablation.py --no_skip_existing  # re-run everything
"""

import subprocess
import os
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
ALL_DATASETS = [
    "as_novel_view",
    "basin_novel_view",
    "bell_novel_view",
    "cup_novel_view",
    "plate_novel_view",
    "press_novel_view",
    "sieve_novel_view",
]

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


def output_dir(dataset, variant_name):
    return os.path.join("output_ablation", dataset, variant_name)


def is_done(dataset):
    """Return True if the dataset folder already exists in output_ablation."""
    return os.path.isdir(os.path.join("output_ablation", dataset))


def make_comparison_png(dataset, variants):
    """Create a bar chart comparing PSNR, SSIM, LPIPS across variants for a dataset."""
    metrics_keys = ["PSNR", "SSIM", "LPIPS"]
    data = {k: [] for k in metrics_keys}
    names = []

    for variant in variants:
        results_path = os.path.join(output_dir(dataset, variant["name"]), "results.json")
        names.append(variant["name"])
        if os.path.exists(results_path):
            try:
                with open(results_path) as f:
                    d = json.load(f)
                for key, val in d.items():
                    if key.startswith("ours_") and isinstance(val, dict):
                        for m in metrics_keys:
                            data[m].append(val.get(m, float("nan")))
                        break
                else:
                    for m in metrics_keys:
                        data[m].append(float("nan"))
            except Exception:
                for m in metrics_keys:
                    data[m].append(float("nan"))
        else:
            for m in metrics_keys:
                data[m].append(float("nan"))

    x = np.arange(len(names))
    n_metrics = len(metrics_keys)
    bar_width = 0.25

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    fig.suptitle(f"Ablation — {dataset}", fontsize=14, fontweight="bold")

    for i, (ax, metric) in enumerate(zip(axes, metrics_keys)):
        values = data[metric]
        bars = ax.bar(x, values, width=bar_width * 2, color=plt.cm.tab10(np.linspace(0, 0.6, len(names))))
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    dataset_dir = os.path.join("output_ablation", dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    out_path = os.path.join(dataset_dir, "ablation_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved comparison: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ODE-GS ablation study runner")
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS,
        metavar="DATASET",
        help="Datasets to evaluate (default: all NeRF-DS datasets)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=ABLATION_NAMES, choices=ABLATION_NAMES,
        metavar="VARIANT",
        help="Ablation variants to run (default: all 5)",
    )
    parser.add_argument(
        "--skip_existing", dest="skip_existing", action="store_true", default=True,
        help="Skip dataset/variant combos where results.json already exists (default: on)",
    )
    parser.add_argument(
        "--no_skip_existing", dest="skip_existing", action="store_false",
        help="Re-run even if results.json already exists",
    )
    args = parser.parse_args()

    selected_variants = [v for v in ABLATIONS if v["name"] in args.variants]
    selected_datasets = args.datasets

    total = len(selected_variants) * len(selected_datasets)
    done  = 0

    print(f"\n{'='*70}")
    print(f"ODE-GS Ablation Study")
    print(f"  Variants : {[v['name'] for v in selected_variants]}")
    print(f"  Datasets : {selected_datasets}")
    print(f"  Total    : {total} experiment(s)")
    print(f"{'='*70}\n")

    for dataset in selected_datasets:
        if args.skip_existing and is_done(dataset):
            print(f"\n  Folder output_ablation/{dataset} exists — skipping dataset.\n")
            done += len(selected_variants)
            continue

        for variant in selected_variants:
            v_name  = variant["name"]
            v_desc  = variant["desc"]
            v_extra = variant["extra_args"]

            done += 1
            out    = output_dir(dataset, v_name)
            source = os.path.join("..", "nerf_ds", dataset)

            print(f"\n{'='*70}")
            print(f"[{done}/{total}]  Dataset: {dataset}  |  Variant: {v_name}")
            print(f"  {v_desc}")
            print(f"  Output : {out}")
            print(f"{'='*70}")

            # ----------------------------------------------------------------
            # 1. Train
            # ----------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # After all variants for this dataset: create comparison PNG
        # -------------------------------------------------------------------
        print(f"\n  Generating ablation_comparison.png for {dataset} ...")
        make_comparison_png(dataset, selected_variants)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")

    col_w = 14
    header = f"{'Dataset':<26}" + "".join(f"{'PSNR ' + v['name']:>{col_w}}" for v in selected_variants)
    print(header)
    print("-" * len(header))

    for dataset in selected_datasets:
        row = f"{dataset:<26}"
        for variant in selected_variants:
            out = output_dir(dataset, variant["name"])
            results_path = os.path.join(out, "results.json")
            psnr_str = "N/A"
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

    print(f"\nOutputs are in:  output_ablation/<dataset>/<variant>/")
    print("All done.\n")


if __name__ == "__main__":
    main()
