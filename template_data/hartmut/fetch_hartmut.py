#!/usr/bin/env python3
"""Fetch + extract the HArtMuT template assets Parrot needs for EEG artifact modeling.

This is Parrot's *fetch-at-first-use* downloader for HArtMuT (Harmening et al. 2022,
GPL-3.0, https://github.com/harmening/HArtMuT). Parrot is MIT-licensed and does **not**
vendor HArtMuT's data; instead this script pulls it on demand into a gitignored cache
(the same philosophy as pulling the pipeline's Docker images). See README.md for the
licensing rationale and provenance.

It runs once per output tree (log-guarded by the orchestrator) and needs network access,
so it is invoked on a networked node — NOT on compute nodes that lack egress (mirror the
hippunfold prewarm pattern; override the destination with HARTMUT_CACHE_HOST to share one
cache across runs).

What it does:
  1. Download the NYhead "small" model (`HArtMuT_NYhead_small.mat`) and the template
     scalp/skull meshes from HArtMuT's GitHub `main`.
  2. Parse the `.mat` once (the only place scipy/.mat parsing happens) and emit clean,
     numpy-only assets that every downstream Parrot container consumes:
       - muscle_sources.npy            (N, 3) float64  muscle source positions, mm, NYhead/MNI frame
       - muscle_labels.npy             (N,)   str      per-source muscle name
       - muscle_leadfield.npy          (n_elec, N, 3)  HArtMuT's precomputed muscle leadfield (fallback)
       - muscle_leadfield_electrodes.csv  label,x,y,z (mm) for the leadfield's 231 channels
       - nyhead_scalp.stl, nyhead_skull.stl            template meshes for the ray-cast warp
       - MANIFEST.json                 provenance + shapes + counts (also the done-marker)

Only *muscle* sources are extracted: Parrot models eyes natively (sampling the subject's own
Eye_balls compartment), so HArtMuT's eye dipoles are intentionally dropped here.
"""
import argparse
import hashlib
import json
import os
import shutil
import urllib.request

import numpy as np
import scipy.io as sio

# HArtMuT GitHub raw endpoints (main). The "small" NYhead model is a plain file (~36 MB);
# only the "large" model is git-LFS, which is why we use "small".
HARTMUT_RAW = "https://raw.githubusercontent.com/harmening/HArtMuT/main"
FILES = {
    "HArtMuT_NYhead_small.mat": f"{HARTMUT_RAW}/HArtMuTmodels/HArtMuT_NYhead_small.mat",
    "nyhead_scalp.stl": f"{HARTMUT_RAW}/individualwarp/NYhead/scalp.stl",
    "nyhead_skull.stl": f"{HARTMUT_RAW}/individualwarp/NYhead/skull.stl",
}
# A muscle source is any artefactmodel entry whose label starts with "Muscle" — this covers
# both the "Muscle_*" and the (rarer) "Muscles_*" spellings in the model, and excludes the
# "Eye*" entries we handle natively.
MUSCLE_PREFIX = "Muscle"


def _download(url, dest, verbose=True):
    """Download url -> dest unless dest already exists (idempotent)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        if verbose:
            print(f"  [skip] {os.path.basename(dest)} already present")
        return
    if verbose:
        print(f"  [get ] {url}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(mat_path, dest, verbose=True):
    """Parse the HArtMuT .mat and write the numpy-only muscle assets into dest."""
    m = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    H = m["HArtMuT"]
    art = H.artefactmodel
    elec = H.electrodes

    pos = np.asarray(art.pos, dtype=np.float64)            # (n_src, 3), mm
    labels = np.array([str(x) for x in np.asarray(art.labels)])
    lf = np.asarray(art.leadfield, dtype=np.float64)       # (n_elec, n_src, 3)
    assert art.unit == "mm", f"unexpected artefactmodel unit: {art.unit!r}"
    assert pos.shape[0] == lf.shape[1] == labels.shape[0], "source-count mismatch"

    muscle = np.array([lab.startswith(MUSCLE_PREFIX) for lab in labels])
    if verbose:
        print(f"  {muscle.sum()} muscle sources of {len(labels)} total "
              f"({len(np.unique(labels[muscle]))} muscle groups)")

    mpos = pos[muscle]
    mlabels = labels[muscle]
    mlf = lf[:, muscle, :]                                 # (n_elec, n_muscle, 3)

    # Electrode geometry for the fallback leadfield's channel dimension.
    chanpos = np.asarray(elec.chanpos, dtype=np.float64)   # (n_elec, 3), mm
    chanlab = np.array([str(x) for x in np.asarray(elec.label)])
    assert elec.unit == "mm", f"unexpected electrode unit: {elec.unit!r}"
    assert chanpos.shape[0] == mlf.shape[0] == chanlab.shape[0], "electrode-count mismatch"

    np.save(os.path.join(dest, "muscle_sources.npy"), mpos)
    np.save(os.path.join(dest, "muscle_labels.npy"), mlabels)
    np.save(os.path.join(dest, "muscle_leadfield.npy"), mlf)
    with open(os.path.join(dest, "muscle_leadfield_electrodes.csv"), "w") as f:
        f.write("label,x,y,z\n")
        for lab, (x, y, z) in zip(chanlab, chanpos):
            f.write(f"{lab},{x:.6f},{y:.6f},{z:.6f}\n")

    return {
        "n_muscle_sources": int(muscle.sum()),
        "n_total_artefact_sources": int(len(labels)),
        "n_muscle_groups": int(len(np.unique(mlabels))),
        "n_electrodes": int(chanpos.shape[0]),
        "muscle_leadfield_shape": list(mlf.shape),
        "source_unit": "mm",
        "source_frame": "NYhead (ICBM-NY, MNI-152 aligned)",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True,
                    help="destination cache dir for downloaded + extracted assets")
    ap.add_argument("--keep-mat", action="store_true",
                    help="keep the raw .mat after extraction (default: delete to save space)")
    args = ap.parse_args()

    dest = args.dest
    os.makedirs(dest, exist_ok=True)

    manifest_path = os.path.join(dest, "MANIFEST.json")
    if os.path.exists(manifest_path):
        print(f"HArtMuT assets already present in {dest} (MANIFEST.json exists); nothing to do.")
        return

    print(f"Fetching HArtMuT assets into {dest} ...")
    for name, url in FILES.items():
        _download(url, os.path.join(dest, name))

    mat_path = os.path.join(dest, "HArtMuT_NYhead_small.mat")
    print("Extracting muscle assets from the .mat ...")
    info = extract(mat_path, dest)

    manifest = {
        "source": "HArtMuT (Harmening et al. 2022), https://github.com/harmening/HArtMuT",
        "license": "GPL-3.0 (upstream data); fetched at use, not vendored — see README.md",
        "model": "HArtMuT_NYhead_small.mat",
        "sha256": {name: _sha256(os.path.join(dest, name))
                   for name in ("nyhead_scalp.stl", "nyhead_skull.stl")},
        **info,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if not args.keep_mat:
        os.remove(mat_path)  # 36 MB; not needed once extracted

    print("Done. Assets:")
    for name in sorted(os.listdir(dest)):
        print("   ", name)


if __name__ == "__main__":
    main()
