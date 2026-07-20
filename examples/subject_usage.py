"""Using parrot_neuro.Subject to read a reconstructed subject's derivatives.

Run it against any Parrot BIDS dataset root:

    pixi run python examples/subject_usage.py /srv/.../parrot_LEMON 010002

Subject is a facade over one subject's derivatives tree. It has two mirrored
namespaces sharing the same vocabulary:

    s.path.<thing>(...)   -> pathlib.Path   (cheap, stdlib-only, file may not exist)
    s.load.<thing>(...)   -> loaded object  (nibabel image / trimesh mesh / ndarray / dict)

For anything without a curated accessor, use the generic escape hatch:

    s.path.sfile(stage, *parts)              # any file under <stage>/sub-<id>/
    s.load.{npy,volume,mesh,json,table}(stage, *parts)
"""
from __future__ import annotations

import sys

from parrot_neuro import Subject


def main(bids_root: str, subject_id: str) -> None:
    # cache=True memoizes loaded objects (handy for big leadfields/meshes in a notebook).
    # warn_on_qc=True (default) warns if you load an output whose stage QC'd as warn/fail.
    s = Subject(bids_root, subject_id, cache=True)
    print(s)  # Subject('010002', deriv=..., backend=fastsurfer)

    # --- subject metadata ---------------------------------------------------
    print("participants.tsv row:", s.participants_row)
    print("overall QC:", (s.qc or {}).get("overall_status"))
    print("leadfields QC status:", s.qc_status("leadfields"))

    # --- discover the variable outputs (globbed, not hardcoded) -------------
    print("leadfields:", s.available_leadfields())
    print("atlas resolutions:", s.atlas_resolutions())
    print("dipole spacings:", s.dipole_spacings())
    print(
        "optional stages:",
        {k: getattr(s, f"has_{k}") for k in ("dwi", "anisotropy", "artifacts", "eeg", "fmri")},
    )

    # --- paths (no I/O) -----------------------------------------------------
    print("T1 path:", s.path.t1())
    print("lh cortex path:", s.path.cortex("lh"))

    # --- load objects -------------------------------------------------------
    t1 = s.load.t1()  # nibabel image -> keeps the affine; call .get_fdata() yourself
    print("T1:", t1.shape, "voxel->world affine:\n", t1.affine)

    key = s.available_leadfields()[0]
    lf = s.load.leadfield(key)  # (n_electrodes, 3*n_sources) ndarray
    print(f"leadfield {key!r}:", lf.shape)

    cortex = s.load.cortex("lh")  # trimesh.Trimesh
    print("lh cortex mesh:", cortex.vertices.shape, cortex.faces.shape)

    dip = s.load.dipoles(s.dipole_spacings()[0])  # DipoleSet bundle
    print("dipoles:", len(dip), "positions", dip.positions.shape)

    electrodes = s.load.electrodes()  # {name: array([x, y, z])}
    print("electrodes:", len(electrodes), "e.g.", next(iter(electrodes.items())))

    # optional stages degrade gracefully -- guard with the has_* flags
    if s.has_dwi:
        print("DTI FA (T1 space):", s.load.dwi_param("fa").shape)
    if s.has_eeg:
        eeg = s.load.eeg("eyesclosed")  # NpzFile of splice-free segments
        print("EEG segments:", list(eeg.keys())[:3], "...")

    # escape hatch for anything without a curated accessor
    agg = s.load.json("atlas", "atlas_to_aggregated.json")
    print("atlas_to_aggregated entries:", len(agg))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: python {sys.argv[0]} <bids_root> <subject_id>")
    main(sys.argv[1], sys.argv[2])
