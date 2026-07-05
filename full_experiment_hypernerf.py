import subprocess
import os

# Each entry is the outer folder name inside hypernerf_interp/.
# The actual dataset lives one level deeper (auto-detected at runtime).
SCENES = [
    "interp_aleks-teapot",
    "interp_chickchicken",
    "interp_cut-lemon",
    "interp_hand",
    "interp_slice-banana",
    "interp_torchocolate",
]

ITERATIONS = 40_000

def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

for scene in SCENES:
    # The dataset folder sits one level inside the outer interp folder.
    # e.g. hypernerf_interp/interp_aleks-teapot/aleks-teapot
    outer_path = os.path.join("..", "hypernerf_interp", scene)
    inner_dirs = [
        d for d in os.listdir(outer_path)
        if os.path.isdir(os.path.join(outer_path, d))
    ]
    assert len(inner_dirs) == 1, (
        f"Expected exactly one dataset subfolder in {outer_path}, got: {inner_dirs}"
    )
    source = os.path.join(outer_path, inner_dirs[0])
    output = os.path.join("output", scene)

    print(f"\n{'='*60}")
    print(f"Scene: {scene}  (source: {source})")
    print(f"{'='*60}")

    if os.path.exists(output):
        print(f"Output folder '{output}' already exists — skipping.")
        continue

    # Train (HyperNeRF interp datasets do not use --is_6dof)
    run(["python", "train.py", "-s", source, "-m", output, "--eval", "--load2gpu_on_the_fly"])

    # Render test set
    run(["python", "render.py", "-m", output, "--skip_train", "--mode", "render"])

    # FFmpeg side-by-side comparison video
    renders_dir = os.path.join(output, "test", f"ours_{ITERATIONS}", "renders", "%05d.png")
    gt_dir      = os.path.join(output, "test", f"ours_{ITERATIONS}", "gt",      "%05d.png")
    out_video   = os.path.join(output, "test", f"ours_{ITERATIONS}", "comparison.mp4")
    run([
        "ffmpeg", "-y",
        "-framerate", "30", "-i", renders_dir,
        "-framerate", "30", "-i", gt_dir,
        "-filter_complex", "hstack,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_video,
    ])

    # Metrics
    run(["python", "metrics.py", "-m", output])

print("\nAll scenes done.")
