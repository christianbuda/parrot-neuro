"""Subject scalp fiducials (NAS/LPA/RPA/IN) from corrected MNI template coordinates.

SimNIBS does not detect fiducials: charm warps four hardcoded MNI points
(``simnibs/resources/ElectrodeCaps_MNI/Fiducials.csv``) through its nonlinear
MNI->subject deformation. Those template points place LPA/RPA at the tragus /
ear-canal rather than at the preauricular depression, which is a systematic
bias every subject inherits.

Measured against 20 hand-picked LEMON subjects, SimNIBS's LPA/RPA are
**7.6 mm posterior** (100% of 40 picks on the same side, SD 2.4 mm) and
~4.7 mm inferior. Besides being anatomically wrong, the posterior bias puts the
preauricular point on the pinna, where the coronal scalp cut used to bisect for
Cz wraps ear cartilage -- inflating that side's arc length and rolling the whole
10-5 montage about the NAS-IN axis (9.3 mm of midline error on AEGEUS P004).

This script replaces the template coordinates with the median of those 20 manual
picks, back-warped into MNI via ``subject2mni_coords``, and warps them into the
subject with the same ``mni2subject_coords`` charm uses. Output is written to
``scalplandmarks/sub-<ID>/fiducials.json``, which ``place_electrodes.py`` prefers
over the SimNIBS CSV when present.

Validated on the 10-subject AEGEUS cohort: Cz offset from the scalp's own
mirror-symmetry plane drops from RMS 3.75 / max 9.47 mm to RMS 0.82 / max 1.45,
and the full 345-electrode mirror error on the two worst subjects falls from
14.86 and 7.43 mm to 1.71 and 1.61 -- better than any originally-healthy subject.

NAS and IN are left at the SimNIBS template values: the manual picks show no
significant bias for either (NAS anterior -0.05 mm; IN unbiased but with 9 mm of
z-scatter, the inion being a notoriously unreliable landmark).

Must run under the SimNIBS interpreter, which is the only one that can import
simnibs in this image:

    /opt/SimNIBS-4.5/simnibs_env/bin/python /scripts/make_fiducials.py \
        --subject <ID> --output_dir /derivatives
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np
from simnibs.utils.transformations import mni2subject_coords

# MNI coordinates. LPA/RPA are the median of 20 manual LEMON picks back-warped to
# MNI; NAS/IN are SimNIBS's own template values. Left/right are deliberately NOT
# symmetrised: the picks reproduce the same ~1.5 mm x-asymmetry the SimNIBS
# template has, so it is a property of the MNI152 head, not an error. Forcing
# symmetry measurably degraded Cz accuracy (RMS 1.09 -> 1.38 mm).
FIDUCIALS_MNI = {
    "LPA": (-78.9, -12.6, -55.2),
    "RPA": (80.3, -12.2, -53.0),
    "NAS": (0.0, 82.9, -43.0),
    "IN": (0.0, -116.2, -30.5),
}
SOURCE = "LEMON manual median (n=20), back-warped to MNI; NAS/IN from SimNIBS template"


def read_ply_vertices(path):
    """Vertices of a binary_little_endian PLY whose first element is vertex x/y/z
    floats. Returns None if the layout is anything else, so callers can skip the
    scalp snap rather than emit garbage coordinates."""
    with open(path, "rb") as fh:
        header = b""
        while b"end_header" not in header:
            line = fh.readline()
            if not line:
                return None
            header += line
        if b"binary_little_endian" not in header:
            return None
        m = re.search(rb"element vertex (\d+)", header)
        if not m:
            return None
        n = int(m.group(1))
        # properties belonging to the vertex element, i.e. up to the next element
        block = header.split(b"element vertex")[1].split(b"element")[0]
        props = re.findall(rb"property\s+(\w+)\s+(\w+)", block)
        if [(t, nm) for t, nm in props] != [(b"float", b"x"), (b"float", b"y"), (b"float", b"z")]:
            return None
        return np.frombuffer(fh.read(n * 12), dtype="<f4").reshape(n, 3).astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--output_dir", required=True, help="derivatives root (e.g. /derivatives)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a fiducials.json this script did not write "
                         "(by default such a file is left alone, so manual picks survive)")
    args = ap.parse_args()

    out_dir = os.path.join(args.output_dir, f"scalplandmarks/sub-{args.subject}")
    os.makedirs(out_dir, exist_ok=True)
    fid_path = os.path.join(out_dir, "fiducials.json")
    prov_path = os.path.join(out_dir, "fiducials_provenance.json")

    # Only ever overwrite our own output. A fiducials.json with no provenance file
    # beside it is a manual pick or a legacy auto-generated one; either way it is
    # not ours to clobber.
    if os.path.isfile(fid_path) and not args.force:
        ours = False
        if os.path.isfile(prov_path):
            try:
                with open(prov_path) as fh:
                    ours = json.load(fh).get("written_by") == "make_fiducials.py"
            except (OSError, ValueError):
                ours = False
        if not ours:
            print(f"fiducials.json already exists and was not written by this script "
                  f"({fid_path}); leaving it alone. Pass --force to replace it.")
            return 0

    m2m = os.path.join(args.output_dir, f"simnibscharm/sub-{args.subject}")
    if not os.path.isdir(m2m):
        print(f"ERROR: no charm output at {m2m}", file=sys.stderr)
        return 1

    # SimNIBS derives the subject id from the folder name, which must be m2m_<ID>.
    # The charm outputs live under sub-<ID>, so point a correctly named symlink at
    # them from a temp dir -- keeps the derivatives tree free of stray entries.
    names = list(FIDUCIALS_MNI)
    print(f"Warping {len(names)} fiducials MNI -> sub-{args.subject} ...")
    tmp = tempfile.mkdtemp(prefix="parrot_fid_")
    try:
        link = os.path.join(tmp, f"m2m_{args.subject}")
        os.symlink(os.path.abspath(m2m), link)
        warped = np.atleast_2d(mni2subject_coords([list(FIDUCIALS_MNI[k]) for k in names], link))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Snap onto the scalp so the fiducials satisfy the invariant they are meant to
    # encode: a point on the head surface. place_electrodes re-projects anyway, so
    # this matters for the published artifact rather than for the montage.
    scalp = os.path.join(args.output_dir, f"surfaces/sub-{args.subject}/charm_scalp.ply")
    verts = read_ply_vertices(scalp) if os.path.isfile(scalp) else None
    snapped = {}
    if verts is None:
        print(f"WARNING: could not read {scalp}; writing unsnapped warped coordinates")
        snap_mm = None
        for k, p in zip(names, warped):
            snapped[k] = [float(v) for v in p]
    else:
        snap_mm = {}
        for k, p in zip(names, warped):
            d = np.linalg.norm(verts - p, axis=1)
            i = int(np.argmin(d))
            snapped[k] = [float(v) for v in verts[i]]
            snap_mm[k] = float(d[i])
        print("  snap distances (mm): " + ", ".join(f"{k}={v:.2f}" for k, v in snap_mm.items()))

    with open(fid_path, "w") as fh:
        json.dump(snapped, fh)
    with open(prov_path, "w") as fh:
        json.dump({"written_by": "make_fiducials.py",
                   "source": SOURCE,
                   "mni_coordinates": {k: list(v) for k, v in FIDUCIALS_MNI.items()},
                   "warp": "simnibs mni2subject_coords (nonlinear, charm MNI2Conform)",
                   "snapped_to": "surfaces/charm_scalp.ply" if verts is not None else None,
                   "snap_distance_mm": snap_mm}, fh, indent=1)

    for k in names:
        print(f"  {k}: {np.round(snapped[k], 2)}")
    print(f"Wrote {fid_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
