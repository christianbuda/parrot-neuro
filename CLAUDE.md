# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**Parrot** is a subject-specific brain *forward-modeling and simulation* pipeline. From a
subject's anatomical MRI it reconstructs detailed head anatomy (cortical, subcortical,
cerebellar, and hippocampal surfaces; tissue label volumes; multi-resolution atlases),
builds a volumetric head model, and solves the bioelectric forward problem to produce a
**leadfield matrix** (n_dipoles × n_electrodes). Each heavy stage runs in its own Docker
container.

The leadfield is the current pipeline endpoint. The intended **next stage** — prototyped in
`development/tvb-optim_EEG/eeg.ipynb`, not yet in the repo proper — is neural-mass simulation
via a **JAX-accelerated TVB**: optimize per-node parameters with TVB-optim, then simulate and
project source activity through the leadfield into synthetic EEG/MEG.

Parrot is standalone. An **acoustic/ultrasound** branch (speed-of-sound, density, attenuation,
nonlinearity tissue maps, produced alongside the electrical conductivities) supports the
**AEGEUS** Horizon Europe project — simulating simultaneous EEG+US to train a reconstruction
network. The US side is an add-on for AEGEUS, not core to the simulation tool.

## Running the Pipeline

```bash
# Full pipeline (BIDS app): reconstruction → forward model → leadfields
./bin/run_reconstruction.sh <bids_dir> <output_dir> participant \
    [--participant-label ID ...] [--threads N] [--gpus all|none|device=0,1] \
    [--spacing-openmeeg 4] [--spacing-duneuro-simnibs 3] [--spacing-duneuro-cgal 2] \
    [--dipole-seed N]
```

`run_reconstruction.sh` is the **single entry point**; it runs the whole pipeline end-to-end.
On startup it pulls any missing Docker images automatically (the old standalone `setup.sh` is
gone), so the first run may spend a while downloading. There is no separate forward-model or
solver driver and no local test suite — all processing happens inside the Docker containers.
Running the pipeline needs no local build; the images are built/published only when maintaining
a container (see `bin/build.sh`).

## Architecture: Four Containers, One Orchestrator

`bin/run_reconstruction.sh` invokes four Parrot images (plus external `deepmi/fastsurfer`
and `khanlab/hippunfold`) in sequence, once per subject:

```
MRI (T1 [+ optional T2/FLAIR, DTI])
    │
    ▼
parrot_mri_reconstruction   (CUDA 12.4 / Ubuntu 22.04)
  FastSurfer · FreeSurfer 8.1 · FSL FIRST · SimNIBS 4.5 (charm) · HippUnfold
  · MNE BEM · Schaefer atlases (100–1000) · cerebellum + BigBrain registration
  → surfaces, tissue label volumes (electrical + acoustic), atlases, electrodes
    │
    ▼  (when DWI present — consumes FastSurfer ACT, the atlas, and raw/T1)
pennlinc/qsiprep · pennlinc/qsirecon   (external; MRtrix3 / ANTs)
  QSIPrep DWI preproc (→ AC-PC space) · QSIRecon SS3T/MSMT tractography (ACT-hsvs)
  · make_dwitensor.sh (DTI fit) · dwi_to_t1.sh (reuse QSIPrep xfm → T1/mesh space)
  · make_connectomes.sh (tck2connectome, T1 space)
  → structural connectome (TVB), diffusion tensor (for WM anisotropy)
    │
    ▼
parrot_forward_model        (CGAL 6.1.1 / Python 3.12)
  place_electrodes.py · place_dipoles.py (Poisson-disk) · nifti_to_inr.py
  · mesher (C++ CGAL) · mesh_postprocessing.py
  → dipole positions/normals, tetrahedral volume mesh
    │
    ▼
parrot_forward_solvers      (DUNEuro 2.10 / OpenMEEG 2.4)
  make_leadfield_openmeeg.py (BEM) · make_leadfield_duneuro.py (FEM)
  → leadfield matrices (one per solver/mesh/spacing)
    │
    ▼
parrot_qc                   (Python 3.12 / nilearn · pyvista offscreen, OSMesa)
  run_qc.py — validate outputs of every stage + render a per-subject report
  (2D label/FA overlays + 3D snapshots: BEM nesting, dipoles, sensitivity, …)
  → qc/sub-<ID>/index.html (+ qc_report.json, figures/), group qc/index.html
    │
    ▼
  [planned] JAX-TVB neural simulation → synthetic EEG/MEG
```

## Key Files to Know

| File | Role |
|------|------|
| `bin/run_reconstruction.sh` | The entry point; ~720-line BIDS-app orchestrator for all stages |
| `bin/images.sh` | **Single source of truth** for the Docker image tags + build contexts (sourced by `run_reconstruction.sh` and `build.sh`) |
| `bin/build.sh` | Builds (and optionally `--push`es) the Parrot images |
| `bin/stage.sh` + `utils/staging/` | Pre-pipeline cohort → Parrot-ready BIDS staging. `bin/stage.sh <cohort> <src> <out>` runs `utils/staging/<cohort>.py` inside the MRI image; `common.py` holds cohort-agnostic helpers (header hygiene, `participants.tsv` writer) |
| `bin/legend_of_files.txt` | **Authoritative map of the output (derivatives) directory layout** |
| `containers/parrot_mri_reconstruction/scripts/` | Reconstruction step scripts (atlas, surfaces, tissue labels, cerebellum, bigbrain, …) |
| `containers/parrot_forward_model/place_dipoles.py` | Poisson-disk dipole sampling + orientation assignment |
| `containers/parrot_forward_model/mesher.cpp` | CGAL tetrahedral mesher (C++) |
| `containers/parrot_forward_solvers/make_leadfield_duneuro.py` | FEM leadfield (DUNEuro) |
| `containers/parrot_forward_solvers/make_leadfield_openmeeg.py` | BEM leadfield (OpenMEEG) |
| `containers/parrot_qc/qc/` | Final QC package (baked into the `parrot_qc` image): `run_qc.py` entry point + one `stages/<name>.py` per pipeline stage + `render2d.py`/`render3d.py` + HTML `templates/` |
| `bin/make_dwitensor.sh` | DTI tensor fit (MRtrix `dwi2tensor`), run inside the QSIRecon image |
| `bin/dwi_to_t1.sh` | Carry DWI tensor + tractogram from QSIPrep's ACPC space into T1/mesh space, reusing QSIPrep's `from-ACPC_to-anat` transform (ANTs resample + world-frame gradient rotation + `dwi2tensor` refit; `tcktransform`) |
| `bin/make_connectomes.sh` | Subject connectome via `tck2connectome` (T1 space), run inside the QSIRecon image |
| `containers/parrot_mri_reconstruction/scripts/prepare_connectivity_atlas.py` | Collapse the Parrot atlas into the connectivity node parcellations |
| `template_data/connectivity/` | Legacy group-average TVB connectivity (100 & 1000 regions). **No longer copied by the pipeline** and NOT in subject units (weights ~1e4x smaller, distances ~2-3x shorter, `1e-06` floor, no `weights_invnodevol`). Superseded by the LEMON group connectome; a no-DWI subject now simply has no `connectivity/`. |
| `containers/parrot_mri_reconstruction/scripts/mni_registration.py` | Subject↔MNI affine (antspyx) bridging the HArtMuT template and the subject for the artifact warp (also feeds the fallback electrode interp) |
| `containers/parrot_forward_model/place_artifact_dipoles.py` | EEG artifact dipoles: eyes (native, sampled in `Eye_balls`) + muscle (HArtMuT template positions warped via `hartmut_warp.py`); writes `artifactsources.json` |
| `containers/parrot_forward_model/hartmut_warp.py` | Ray-cast layer-normalized source warp — clean-room port of HArtMuT `project_points.jl` (skull↔scalp depth-fraction preservation) |
| `containers/parrot_forward_solvers/make_leadfield_artifacts.py` | Geometry-only artifact leadfields (eyes + muscle share ONE DUNEuro transfer matrix) |
| `containers/parrot_forward_solvers/make_leadfield_hartmut_muscle.py` | Fallback muscle leadfield (HArtMuT's canned leadfield interpolated onto the subject montage) |
| `template_data/hartmut/` | HArtMuT fetch-at-use downloader (`fetch_hartmut.py`) + README — GPL-3.0 assets fetched, not vendored |

## Conventions & Gotchas

- **Idempotency via log files.** Each step is skipped if its `*_log.txt` already exists under
  `<output_dir>/logs/sub-<ID>/`. To force a rerun, delete the relevant log file (not the
  outputs). Stale logs are the usual cause of "why was this step skipped?".
- **BIDS I/O.** Input is a BIDS dataset; T1 is required, T2/FLAIR optional (auto-discovered,
  used for pial refinement / charm). Output is a BIDS-derivatives tree — see
  `bin/legend_of_files.txt` for the full layout.
- **Per-subject overrides** come from `participants.tsv` columns (e.g. skip-T2-registration,
  no-neck). Parsing is positional, so column order matters.
- **Solver/spacing pairing.** Three leadfields are computed at different dipole spacings:
  OpenMEEG/BEM (`--spacing-openmeeg`, 4 mm), DUNEuro/FEM on the SimNIBS charm mesh
  (`--spacing-duneuro-simnibs`, 3 mm), DUNEuro/FEM on the CGAL mesh (`--spacing-duneuro-cgal`,
  2 mm). Dipoles are pre-sampled at all three spacings.
- **sim4life is optional/manual.** If a `sim4life.nii.gz` tissue volume exists it is preferred
  for meshing; otherwise the SimNIBS labels are used.
- **GPU auto-detection.** `--gpus` falls back to CPU-only if `nvidia-smi` is missing.
- **DWI stages (optional, external images).** QSIPrep/QSIRecon (`pennlinc/*`) run only when DWI
  is present. The `bin/make_*.sh` and `bin/dwi_to_t1.sh` scripts run *inside* the QSIRecon image
  (MRtrix3 + ANTs) and are bind-mounted, so editing them needs no image rebuild. Gotchas:
  QSIPrep gets a **raw-only `/bids`** (subject + dataset metadata only — its pybids indexer
  crashes on the nested derivatives tree); MRtrix **cannot read `.tck.gz`** (decompress first);
  and QSIPrep works in **AC-PC space**, which is *not* the mesh/T1 space — `dwi2t1` reuses
  QSIPrep's own `from-ACPC_to-anat` transform (its "anat" == `raw/T1`), applied with ANTs, to
  reach mesh space. We do **not** re-register (`mrregister` diverged; reuse is correct + robust).
- **Final QC (`parrot_qc`).** A final stage validates the outputs of every reconstruction stage
  and renders a human-review report. It runs **automatically** as the last per-subject step, then
  a group pass after the subject loop. Unlike every other step it is **not** log-guarded — it
  **always runs** (so the report reflects the latest outputs) and is **non-fatal** (a QC failure
  only logs a WARNING; it never aborts a reconstruction). It's quick (~2 min/subject) and runs
  rootless, reading only `/derivatives`.
  - **View:** open `<output_dir>/qc/sub-<ID>/index.html` (per-subject report: pass/warn/fail/skip
    table + embedded figures); the group `subject × stage` matrix is `<output_dir>/qc/index.html`;
    `qc/sub-<ID>/qc_report.json` is the machine-readable record. `skip` = stage not produced
    (optional stages like DWI degrade gracefully, never fail).
  - **Run standalone** (inside the `parrot_qc` image, mounting derivatives at `/derivatives`):
    `python /qc/run_qc.py --subject <ID> --output_dir /derivatives` (or `--group`).
  - **Edit/iterate:** the QC code is the baked `qc/` package (like the other images' scripts), so
    changes need a rebuild: `./bin/build.sh qc`. Add a stage by dropping a `stages/<name>.py`
    (exposing `NAME`, `TITLE`, `run(ctx)`) into the ordered list in `qc/stages/__init__.py`.
- **EEG artifacts stage (`artifacts`).** Adds extra-brain eye/muscle noise sources → geometry-only
  artifact leadfields (see the roadmap entry + `legend_of_files.txt`). Runs per subject after the
  solvers, spans three images (registration in `parrot_mri_reconstruction`; dipoles in
  `parrot_forward_model`; leadfields in `parrot_forward_solvers`), FEM/CGAL only. Gotchas:
  - **Fetched, not vendored.** HArtMuT is GPL-3.0; a one-time pre-loop step downloads its assets to
    `<output_dir>/.hartmut_cache` and the MNI152NLin2009cAsym T1w to the templateflow cache. This
    **needs network egress** — on egress-less HPC nodes prewarm off-cluster (set `HARTMUT_CACHE_HOST`
    to a prepared cache and pre-place the MNI template), like the hippunfold prewarm. If setup can't
    complete, the per-subject `artifacts` stage **skips gracefully** (non-fatal).
  - **Muscle solve vs fallback.** `place_artifact_dipoles.py` warps HArtMuT muscle positions onto the
    subject; if too few survive (no neck FOV → `neck_coverage:false` in `artifactsources.json`), the
    orchestrator uses HArtMuT's canned muscle leadfield interpolated onto the montage instead of solving.
  - **Eyes + muscle share ONE transfer matrix** (`make_leadfield_artifacts.py`) — the expensive
    DUNEuro step depends only on mesh/conductivities/electrodes, not the dipoles.
  - **QC:** the `parrot_qc` stage `artifacts` validates the registration/dipole/leadfield outputs and
    renders source positions + sample EOG/EMG topographies (it `skip`s when the stage didn't run).
  - **New deps/rebuilds:** `parrot_forward_model` needs `embreex`+`rtree` (fast ray-casting) and the
    new scripts baked; `parrot_mri_reconstruction` needs `mni_registration.py` baked; `parrot_qc`
    needs the new `stages/artifacts.py` baked. Rebuild those three images (+ `parrot_forward_solvers`
    for the new solver scripts) after pulling this feature.

## Repo Layout Notes

- `src/parrot_neuro/`, `tests/`, `examples/`, `external/` are **scaffolding, currently
  empty** — the README's "Python API" is aspirational, not yet implemented.
- `utils/staging/` holds the dataset-staging tooling (see Key Files); the rest of `utils/`
  is still scaffolding. Run staging via `bin/stage.sh`, never on the host (needs nibabel).
- `development/` is git-ignored scratch (JAX-TVB, tractography, US, EEG prototypes): where the
  next stages are prototyped, not part of the shipped pipeline.
- Docker images are published on Docker Hub under `christianbuda/`. Build/publish them with
  `bin/build.sh` (e.g. `./bin/build.sh --push`, or `./bin/build.sh solvers` for one image);
  the image list and build contexts live once in `bin/images.sh`.

## Planned / In-Progress Directions

Branch `feat/dwi-connectivity-anisotropy-hartmut` and beyond — keep these in mind when planning:

- DWI preprocessing + tractography → subject-specific structural connectivity. **[done]**
  (QSIPrep → QSIRecon → `connectivity`; connectome built in T1/mesh space).
- Anisotropic conductivity tensors (from DWI) for the FEM leadfield. **[in progress]** DTI fit
  (`dwitensor`) and registration into mesh space (`dwi2t1`) are done; remaining: map eigenvalues →
  per-element conductivity tensors (shape-preserving orthotropic) and feed full 3×3 tensors to
  DUNEuro in `make_leadfield_duneuro.py` (which currently uses isotropic per-label conductivities).
- EEG noise modeling (extra-brain artifact sources). **[done — source geometry/leadfields]**
  HArtMuT-informed eye (native, sampled in the subject `Eye_balls`) + face/neck muscle (template
  positions warped into the subject via a subject↔MNI affine + a ray-cast layer-normalized
  projection) artifact sources → **geometry-only** DUNEuro artifact leadfields, stackable with the
  brain leadfield (`artifacts` stage; FEM/CGAL only). HArtMuT assets are fetched-at-use (GPL-3.0,
  not vendored — `template_data/hartmut/`). Remaining: the amplitude/noise generator (MUAP-EMG +
  gaze/blink EOG time series projected through these leadfields) — a future stage.
- JAX-accelerated TVB simulation stage with TVB-optim parameter fitting.
