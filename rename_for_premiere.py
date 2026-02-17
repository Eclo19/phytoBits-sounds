#!/usr/bin/env python3
"""
Robust in-place renamer for Premiere Image Sequence import.

- Looks for timestamps like YYYYMMDD_HHMMSS anywhere in the filename.
  Example: main_photo_20260204_210239.jpg
           main_photo_20260204_210239
           something_20260204_210239_extra.JPG

- Sorts by that timestamp, then renames to:
    frame_000001.jpg, frame_000002.jpg, ...

USAGE:
  python3 rename_for_premiere.py "/path/to/Plant Loop Frames"

NOTE:
  If a file has no extension, it will use DEFAULT_EXT below (jpg).
"""

import re
import sys
from pathlib import Path
from datetime import datetime

# If a file has no suffix, use this extension:
DEFAULT_EXT = ".jpg"

# Find YYYYMMDD_HHMMSS anywhere in the name (with optional separators around it)
TS_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})")

def extract_ts(filename: str) -> datetime | None:
    m = TS_RE.search(filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")
    except ValueError:
        return None

def main(folder: str) -> int:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: Not a folder: {root}")
        return 2

    candidates = []
    for p in root.iterdir():
        if not p.is_file():
            continue
        ts = extract_ts(p.name)
        if ts is None:
            continue
        candidates.append((ts, p))

    if not candidates:
        print("ERROR: No files with timestamps like YYYYMMDD_HHMMSS were found.")
        return 1

    # Sort by time, then by name to stabilize ties
    candidates.sort(key=lambda x: (x[0], x[1].name.lower()))

    n = len(candidates)
    pad = max(6, len(str(n)))

    # Build target names
    originals = {p.resolve() for _, p in candidates}
    targets = []
    for i, (_, p) in enumerate(candidates, start=1):
        ext = p.suffix.lower() if p.suffix else DEFAULT_EXT
        new_name = f"frame_{i:0{pad}d}{ext}"
        targets.append((p, root / new_name))

    # Conflict check: if target exists and isn't one of the originals
    for old_p, new_p in targets:
        if new_p.exists() and new_p.resolve() not in originals:
            print(f"ERROR: Target already exists and is not part of this rename set: {new_p.name}")
            print("Move/delete that file or choose a different folder/name scheme.")
            return 3

    print(f"Found {n} timestamped files. Renaming in: {root}")
    print(f"Output pattern: frame_{{index:0{pad}d}}<ext>")

    # PASS 1: rename everything to unique temp names (prevents collisions)
    temp_paths = []
    for i, (_, p) in enumerate(candidates, start=1):
        ext = p.suffix if p.suffix else ""
        tmp = root / f"__tmp__premiere__{i:0{pad}d}__{p.stem}{ext}"
        p.rename(tmp)
        temp_paths.append(tmp)

    # PASS 2: rename temps to final sequential names
    for i, tmp in enumerate(temp_paths, start=1):
        ext = tmp.suffix.lower() if tmp.suffix else DEFAULT_EXT
        final = root / f"frame_{i:0{pad}d}{ext}"
        tmp.rename(final)

    print("Done.")
    print("Premiere: File → Import → click frame_000001... → check “Image Sequence”.")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python3 rename_for_premiere.py "/path/to/Plant Loop Frames"')
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
