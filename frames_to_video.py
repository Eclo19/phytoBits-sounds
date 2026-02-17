#!/usr/bin/env python3
"""
Build a no-audio video from sequential frames like:
  frame_000001.jpg, frame_000002.jpg, ...

Default: 24 fps, output.mp4

USAGE:
  python3 frames_to_video.py "/path/to/Plant Loop Frames"
  python3 frames_to_video.py "/path/to/Plant Loop Frames" --fps 30 --out plant.mp4

Requires: ffmpeg on PATH
"""

import argparse
import re
import subprocess
from pathlib import Path
import sys

FRAME_RE = re.compile(r"^frame_(\d+)(\.[A-Za-z0-9]+)?$")

def detect_sequence(folder: Path):
    # Collect files that match frame_<digits><ext?>
    hits = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = FRAME_RE.match(p.stem + p.suffix) if p.suffix else FRAME_RE.match(p.name)
        if not m:
            continue
        num_str = m.group(1)
        ext = (p.suffix or "").lower()
        hits.append((int(num_str), len(num_str), ext, p.name))

    if not hits:
        return None

    # Choose the most common padding length, then the most common extension within that
    from collections import Counter
    pad_counts = Counter(pad for _, pad, _, _ in hits)
    pad = pad_counts.most_common(1)[0][0]

    ext_counts = Counter(ext for _, pad_i, ext, _ in hits if pad_i == pad)
    ext = ext_counts.most_common(1)[0][0]

    # Ensure we have frame_000001 style numbering starting at 1 (not strictly required by ffmpeg,
    # but it helps detect obvious problems)
    nums = sorted(n for n, pad_i, ext_i, _ in hits if pad_i == pad and ext_i == ext)
    return pad, ext, nums

def run(cmd):
    print("Running:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder containing frame_000001.jpg ...")
    ap.add_argument("--fps", type=float, default=24.0, help="Frames per second (default: 24)")
    ap.add_argument("--out", default="output.mp4", help="Output video filename (default: output.mp4)")
    ap.add_argument("--crf", type=int, default=18, help="Quality (lower=better). Default 18")
    ap.add_argument("--preset", default="slow", help="ffmpeg x264 preset (default: slow)")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"ERROR: Not a folder: {folder}")
        return 2

    detected = detect_sequence(folder)
    if not detected:
        print("ERROR: No files matching frame_<digits>.<ext> were found.")
        return 1

    pad, ext, nums = detected
    if not nums:
        print("ERROR: Sequence detected but no frame numbers found.")
        return 1

    # Pattern ffmpeg expects
    # Note: ext includes the dot, e.g. ".jpg"
    pattern = f"frame_%0{pad}d{ext}"
    out_path = (folder / args.out).resolve()

    # Build ffmpeg command:
    # -framerate sets how to interpret the input sequence timing
    # -r sets output fps (keeps things consistent)
    # -an disables audio
    # -pix_fmt yuv420p for broad compatibility
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "info",
        "-framerate", str(args.fps),
        "-i", str(folder / pattern),
        "-c:v", "libx264",
        "-preset", args.preset,
        "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        "-an",
        "-r", str(args.fps),
        str(out_path),
    ]

    try:
        run(cmd)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install it and make sure it's on your PATH.")
        return 127
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffmpeg failed with exit code {e.returncode}")
        return e.returncode

    print(f"Done. Wrote: {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
