#!/usr/bin/env python3
"""Recursively add binary STL files beside PLY meshes; never overwrite files.

Run from the repository root:
    pixi run python utils/ply_to_stl.py <mesh_directory> --dry-run
    pixi run python utils/ply_to_stl.py <mesh_directory>
"""

import argparse
from pathlib import Path

import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory to search recursively for PLY files",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list planned conversions")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Not a directory: {args.root}")

    converted = skipped = failed = 0
    for source in sorted(args.root.rglob("*.ply")):
        target = source.with_suffix(".stl")
        if target.exists():
            print(f"SKIP (exists): {target}")
            skipped += 1
            continue
        if args.dry_run:
            print(f"WOULD CONVERT: {source} -> {target.name}")
            converted += 1
            continue
        try:
            mesh = trimesh.load(source, force="mesh", process=False)
            if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
                raise ValueError("PLY does not contain a triangle mesh")
            data = mesh.export(file_type="stl")
            # Exclusive creation also prevents overwriting in a concurrent run.
            with target.open("xb") as output:
                output.write(data)
            print(f"CONVERTED: {target}")
            converted += 1
        except FileExistsError:
            print(f"SKIP (exists): {target}")
            skipped += 1
        except Exception as exc:
            print(f"FAILED: {source}: {exc}")
            failed += 1

    label = "Planned" if args.dry_run else "Converted"
    print(f"{label}: {converted}; skipped: {skipped}; failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
