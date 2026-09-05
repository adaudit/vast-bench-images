#!/usr/bin/env python3
"""Bin-pack a virtualenv into fixed-size directory buckets so every image layer stays small.

Usage: split_venv_layers.py VENV OUT_DIR BUCKET_MB MAX_BUCKETS
Writes OUT_DIR/venv-NN/<venv path> trees (hardlinks) covering every file exactly once, and prints a manifest.
Large shared-library files (e.g. libtorch_cuda.so) are packed individually so one file never forces a huge bucket.
"""
import os
import sys
from pathlib import Path


def tree_size(path):
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    return sum(p.lstat().st_size for p in path.rglob("*") if p.is_file() or p.is_symlink())


def units(venv, split_threshold):
    """Yield (relative_path, size) units: top-level site-packages entries, split further when oversized."""
    site = venv / "lib" / "python3.11" / "site-packages"
    for entry in sorted(venv.iterdir()):
        if entry.resolve() != site.parent.resolve() and entry.name != "lib":
            yield entry.relative_to(venv), tree_size(entry)
    for top in sorted(venv.glob("lib/python3.11/site-packages/*")):
        size = tree_size(top)
        if size <= split_threshold or top.is_file() or top.is_symlink():
            yield top.relative_to(venv), size
            continue
        for child in sorted(top.iterdir()):
            child_size = tree_size(child)
            if child_size <= split_threshold or child.is_file() or child.is_symlink():
                yield child.relative_to(venv), child_size
            else:
                for leaf in sorted(child.rglob("*")):
                    if leaf.is_file() or leaf.is_symlink():
                        yield leaf.relative_to(venv), leaf.lstat().st_size
    for other in sorted(venv.glob("lib/*")):
        if other.name != "python3.11":
            yield other.relative_to(venv), tree_size(other)
    for other in sorted(venv.glob("lib/python3.11/*")):
        if other.name != "site-packages":
            yield other.relative_to(venv), tree_size(other)


def link_tree(src, dst):
    if src.is_symlink() or src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            os.symlink(os.readlink(src), dst)
        else:
            os.link(src, dst)
        return
    for path in src.rglob("*"):
        target = dst / path.relative_to(src)
        if path.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.readlink(path), target)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
        elif path.is_dir():
            target.mkdir(parents=True, exist_ok=True)


def main():
    venv, out, bucket_mb, max_buckets = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    limit = bucket_mb * 1024 * 1024
    items = sorted(units(venv, limit // 2), key=lambda item: -item[1])
    buckets = []  # [total, [paths]]
    for rel, size in items:  # first-fit decreasing
        for bucket in buckets:
            if bucket[0] + size <= limit:
                bucket[0] += size
                bucket[1].append(rel)
                break
        else:
            buckets.append([size, [rel]])
    if len(buckets) > max_buckets:
        raise SystemExit(f"{len(buckets)} buckets exceed MAX_BUCKETS={max_buckets}")
    venv_rel = venv.relative_to("/")
    for index, (total, paths) in enumerate(buckets):
        root = out / f"venv-{index:02d}" / venv_rel
        for rel in paths:
            link_tree(venv / rel, root / rel)
        print(f"venv-{index:02d} {total / 1e6:8.1f} MB {len(paths):4d} entries; largest: {paths[0]}")
    for index in range(len(buckets), max_buckets):  # keep COPY instructions static
        (out / f"venv-{index:02d}" / venv_rel).mkdir(parents=True, exist_ok=True)
    print(f"buckets={len(buckets)} total_MB={sum(b[0] for b in buckets) / 1e6:.0f}")


if __name__ == "__main__":
    main()
