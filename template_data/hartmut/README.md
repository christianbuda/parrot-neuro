# HArtMuT template assets (fetched, not vendored)

Parrot's EEG **artifact** modeling (extra-brain eye/muscle noise sources) is informed by
**HArtMuT** — the *Head Artefact Model using Tripoles* (Harmening, Klug, Gramann & Miklody,
2022; https://github.com/harmening/HArtMuT). HArtMuT models eyes and face/neck muscles as
extra-brain EEG sources and ships a template artifact source set + precomputed leadfield on
the NYhead (ICBM-NY) geometry.

## Why this directory only holds a script

HArtMuT's data is **GPL-3.0**; Parrot is **MIT**. To avoid mixing GPL-3.0 data into an MIT
source tree (and to keep large binaries out of git history), Parrot does **not** vendor
HArtMuT. Instead `fetch_hartmut.py` pulls the assets **at first use** into a gitignored
cache — the same philosophy as pulling the pipeline's Docker images on first run.

Only a clean-room **reimplementation** of HArtMuT's ray-cast layer-normalized warp algorithm
lives in Parrot's code (`containers/parrot_forward_model/hartmut_warp.py`); no GPL-licensed
code is copied. Algorithms are not copyrightable — the data is fetched, the method is
reimplemented.

## What gets fetched

`fetch_hartmut.py --dest <cache>` downloads from HArtMuT `main` and extracts numpy-only assets:

| Asset | What it is |
|-------|------------|
| `muscle_sources.npy` `(N,3)` | muscle source positions, mm, NYhead/MNI frame (eyes dropped — Parrot models them natively) |
| `muscle_labels.npy` `(N,)` | per-source muscle name (e.g. `Muscle_Temporalis_...`) |
| `muscle_leadfield.npy` `(n_elec,N,3)` | HArtMuT's precomputed muscle leadfield (the fallback path) |
| `muscle_leadfield_electrodes.csv` | `label,x,y,z` (mm) for the leadfield's 231 channels |
| `nyhead_scalp.stl`, `nyhead_skull.stl` | template meshes for the ray-cast warp |
| `MANIFEST.json` | provenance, shapes, counts, mesh checksums (also the done-marker) |

From the NYhead "small" model: **3180 muscle sources** across **53 muscle groups**, leadfield
`(231, 3180, 3)`. The parse of the `.mat` happens **only here** (the one place scipy touches
it); every downstream container consumes the numpy assets.

## How it is invoked

The orchestrator runs it (log-guarded) into an output-side cache
(`$OUTPUT_DIR/.hartmut_cache`, override with `HARTMUT_CACHE_HOST` to share one cache across
runs), mirroring the hippunfold prewarm pattern. It needs network egress, so on HPC run it on
a networked node — not on compute nodes without internet.

## Citation

Harmening N., Klug M., Gramann K., Miklody D. (2022). *HArtMuT — Modeling eye and muscle
contributors in neuroelectric imaging.* Journal of Neural Engineering.
