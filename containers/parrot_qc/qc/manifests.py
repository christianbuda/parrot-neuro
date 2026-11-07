"""Register selected outputs at producer completion, never by scanning old logs.

The writer still owns filename knowledge. Context supplies the shared QC stage
layout; the contract tests keep these selections aligned with Subject accessors.
"""
from __future__ import annotations

import argparse
import json
import os
import re

from .context import Context


def write_manifest(deriv, subject, step, producer_tag, *, spacing=None, task=None,
                   paths=None, variant=None):
    """Write one completed step. ``paths`` restricts staging to newly produced files."""
    if not producer_tag:
        raise ValueError("The actual producer Docker image tag is required")
    ctx = Context(str(deriv), str(subject).removeprefix("sub-"))
    root = ctx.deriv
    outputs = []

    def add(path, kind, **parameters):
        if path.is_file() and (paths is None or path in paths):
            entry = {"type": kind, "parameters": parameters,
                     "path": path.relative_to(root).as_posix()}
            if kind == "leadfield":
                entry["row_order"] = "electrodes CSV order (full montage)"
                entry["column_order"] = "dipole-major xyz"
            outputs.append(entry)

    def files(stage, patterns):
        directory = ctx.stage_dir(stage)
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                parameters = {"name": path.relative_to(directory).as_posix()}
                if stage == "dipoles":
                    parameters["spacing"] = float(spacing)
                if stage in {"EEG", "fMRI"}:
                    parameters["task"] = task
                    if variant is not None:
                        parameters["variant"] = variant
                match = re.search(r"(?:^|atlas|_)(\d+)(?:[_.]|Parcels)", path.name)
                if match:
                    parameters["resolution"] = int(match[1])
                add(path, stage, **parameters)

    simple = {
        "ingest": ("raw", ["T1.nii.gz", "T2.nii.gz", "T*.json"]),
        "synthstrip": ("synthstrip", ["T1_stripped*.nii.gz"]),
        "simnibscharm": ("simnibscharm", ["final_tissues.nii.gz", "subject.msh"]),
        "surfaces": ("surfaces", ["*.ply", "*.npy"]),
        "atlas": ("atlas", ["atlas*.nii.gz", "atlas*_LUT.txt", "atlas*_labels.txt", "atlas_to_aggregated.json"]),
        "tissuelabels": ("tissuelabels", ["*/*.nii.gz", "*/*.txt"]),
        "fiducials": ("scalplandmarks", ["fiducials.json"]),
        "electrodes": ("electrodes", ["landmarks_10-5-full.csv", "selected_landmarks_10-5-full.json"]),
        "tetmesh": ("tetmesh", ["tetrahedral_mesh.mesh", "tetrahedral_mesh.vtu"]),
        "connectivity-atlas": ("connectivity", ["atlas*.nii.gz", "labels_*.txt", "*_to_*.npy"]),
        "connectivity": ("connectivity", ["weights*.txt", "distances*.txt"]),
        "dwitensor": ("dwitensor", [f"*space-{'T1' if task == 'hcp' else 'ACPC'}*.nii.gz"]),
        "dwi2t1": ("dwitensor", ["*space-T1*.nii.gz", "*_ras.txt"]),
        "anisotropy": ("anisotropy", ["*.npy"]),
        "fastsurfer": ("fastsurfer", ["surf/lh.white"]),
        "freesurfer": ("freesurfer", ["surf/lh.white"]),
        "recon-lut-fastsurfer": ("fastsurfer", ["FreeSurferColorLUT.txt"]),
        "recon-lut-freesurfer": ("freesurfer", ["FreeSurferColorLUT.txt"]),
        "qsirecon": ("qsirecon", ["dwi/*streamlines.tck.gz"]),
    }
    if step in simple:
        files(*simple[step])
        if step == "dwi2t1":
            files("qsirecon", ["dwi/*space-T1_*streamlines.tck.gz"])
    elif step == "dipoles":
        spacing = float(spacing)
        files("dipoles", [f"spacing{spacing}mm/*.npy", f"spacing{spacing}mm/volumetric/*.npy",
                          f"spacing{spacing}mm/surfaces/*/*.npy"])
        step = f"dipoles-{spacing}mm"
    elif step.startswith("forwardsolvers-"):
        solver = step.removeprefix("forwardsolvers-")
        label = {"openmeeg": "openmeeg", "simnibs": "duneurosimnibs",
                 "cgal": "duneuroCGAL", "anisotropic": "duneuroCGAL_anisotropic"}[solver]
        key = f"{label}-{float(spacing)}mm"
        add(ctx.sfile("leadfields", f"processed_{key}-leadfield.npy"), "leadfield", key=key)
    elif step in {"artifacts-registration", "artifacts-dipoles", "artifacts-leadfields"}:
        if step == "artifacts-leadfields":
            if variant not in {"eyes", "muscle", "muscle-fallback", "eyes-positions", "muscle-positions"}:
                raise ValueError("Artifact leadfields require an explicit group/positions variant")
            if variant.endswith("-positions"):
                group = variant.removesuffix("-positions")
                path = root / "artifacts" / "dipoles" / ctx.subj / group / "dipole_positions_solved.npy"
                add(path, "artifacts", name=path.relative_to(root / "artifacts").as_posix())
            else:
                key = ("hartmut_muscle" if variant == "muscle-fallback"
                       else f"duneuro_artifact-{variant}-CGAL")
                path = ctx.sfile("leadfields", f"processed_{key}-leadfield.npy")
                add(path, "leadfield", key=key)
                add(path.with_suffix(".json"), "leadfield_metadata", key=key)
            step += f"-{variant.removesuffix('-fallback')}"
        else:
            part = step.removeprefix("artifacts-")
            directory = root / "artifacts" / part / ctx.subj
            patterns = (["*_affine.npy", "registration_qc.json"] if part == "registration" else
                        ["artifactsources.json", "*/dipole_positions.npy", "*/dipole_volume.npy",
                         "*/dipole_preferential_direction.npy", "*/dipole_labels.npy",
                         "*/orient_type.npy", "*/template_index.npy"])
            for pattern in patterns:
                for path in sorted(directory.glob(pattern)):
                    add(path, "artifacts", name=path.relative_to(root / "artifacts").as_posix())
    elif step in {"eeg", "fmri", "fmri-nodes", "fmri-transform", "fmri-bold"}:
        stage = "EEG" if step == "eeg" else "fMRI"
        files(stage, [f"{ctx.subj}_task-{task}_*"] if step != "fmri-transform"
              else [f"{ctx.subj}_from-native_to-T1w_xfm.mat"])
        step = f"{step}-{task}"
        if variant is not None:
            step += f"-{variant}"
    else:
        raise ValueError(f"Unknown manifest step: {step}")
    destination = root / "manifests" / ctx.subj / f"{step}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a killed process never leaves a truncated manifest behind.
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_text(json.dumps({"producer_tag": producer_tag, "outputs": outputs}, indent=2) + "\n")
    os.replace(tmp, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--producer-tag", required=True)
    parser.add_argument("--spacing", type=float)
    parser.add_argument("--task")
    parser.add_argument("--variant")
    args = parser.parse_args()
    write_manifest(args.output_dir, args.subject, args.step, args.producer_tag,
                   spacing=args.spacing, task=args.task, variant=args.variant)


if __name__ == "__main__":
    main()
