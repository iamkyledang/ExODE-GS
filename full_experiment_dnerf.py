import subprocess
import os

# D-NeRF synthetic Blender scenes
SCENES = [
    "bouncingballs",
    "hellwarrior",
    "hook",
    "jumpingjacks",
    "lego",
    "mutant",
    "standup",
    "trex",
]

ITERATIONS = 40_000

def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

for scene in SCENES:
    # dnerf scenes are direct subfolders (no extra nesting unlike hypernerf)
    source = os.path.join("..", "dnerf", scene)
    output = os.path.join("output_dnerf", scene)

    print(f"\n{'='*60}")
    print(f"Scene: {scene}  (source: {source})")
    print(f"{'='*60}")

    if os.path.exists(output):
        print(f"Output folder '{output}' already exists — skipping.")
        continue

    # Train
    # D-NeRF is a synthetic Blender dataset:
    #   - scene/__init__.py auto-detects Blender via transforms_train.json
    #   - white_background=True  (rendered on white bg)
    #   - no --is_6dof  (standard translation-only deformation)
    #   - load2gpu_on_the_fly to handle larger scenes
    run([
        "python", "train.py",
        "-s", source,
        "-m", output,
        "--eval",
        "--white_background",
        "--load2gpu_on_the_fly",
    ])

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

print("\nAll D-NeRF scenes done.")
