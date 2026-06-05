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

## Architecture: Three Containers, One Orchestrator

`bin/run_reconstruction.sh` invokes three Parrot images (plus external `deepmi/fastsurfer`
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
  [planned] JAX-TVB neural simulation → synthetic EEG/MEG
```

## Key Files to Know

| File | Role |
|------|------|
| `bin/run_reconstruction.sh` | The entry point; ~720-line BIDS-app orchestrator for all stages |
| `bin/images.sh` | **Single source of truth** for the Docker image tags + build contexts (sourced by `run_reconstruction.sh` and `build.sh`) |
| `bin/build.sh` | Builds (and optionally `--push`es) the Parrot images |
| `bin/legend_of_files.txt` | **Authoritative map of the output (derivatives) directory layout** |
| `containers/parrot_mri_reconstruction/scripts/` | Reconstruction step scripts (atlas, surfaces, tissue labels, cerebellum, bigbrain, …) |
| `containers/parrot_forward_model/place_dipoles.py` | Poisson-disk dipole sampling + orientation assignment |
| `containers/parrot_forward_model/mesher.cpp` | CGAL tetrahedral mesher (C++) |
| `containers/parrot_forward_solvers/make_leadfield_duneuro.py` | FEM leadfield (DUNEuro) |
| `containers/parrot_forward_solvers/make_leadfield_openmeeg.py` | BEM leadfield (OpenMEEG) |
| `template_data/connectivity/` | Group-average TVB connectivity (100 & 1000 regions); inputs to the planned sim stage |

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

## Repo Layout Notes

- `src/parrot_neuro/`, `tests/`, `utils/`, `examples/`, `external/` are **scaffolding,
  currently empty** — the README's "Python API" is aspirational, not yet implemented.
- `development/` is git-ignored scratch (JAX-TVB, tractography, US, EEG prototypes): where the
  next stages are prototyped, not part of the shipped pipeline.
- Docker images are published on Docker Hub under `christianbuda/`. Build/publish them with
  `bin/build.sh` (e.g. `./bin/build.sh --push`, or `./bin/build.sh solvers` for one image);
  the image list and build contexts live once in `bin/images.sh`.

## Planned / In-Progress Directions

Branch `feat/dwi-connectivity-anisotropy-hartmut` and beyond — keep these in mind when planning:

- DWI preprocessing + tractography → subject-specific structural connectivity.
- Anisotropic conductivity tensors (from DWI) for the FEM leadfield.
- EEG noise modeling.
- JAX-accelerated TVB simulation stage with TVB-optim parameter fitting.
