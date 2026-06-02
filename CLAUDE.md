# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**Parrot** is an end-to-end brain simulation pipeline. It takes a subject's anatomical MRI and produces simulated EEG/MEG signals by chaining four stages: MRI reconstruction → forward mesh modeling → leadfield computation → neural simulation (TVB). Each stage runs in its own Docker container.

## Running the Pipeline

```bash
# One-time: pull Docker images
./bin/setup.sh

# Reconstruction Stage (BIDS-app variant)
./bin/run_reconstruction.sh <bids_dir> <output_dir> participant [--participant-label ID] [--threads N] [--gpus all|none]
```

There is no build step, Makefile, or test suite to run locally — all processing runs inside Docker containers invoked by the shell scripts above.

## Architecture: Four Containers, One Data Flow

```
MRI (T1/T2)
    │
    ▼
parrot_mri_reconstruction   (CUDA 12.4 / Ubuntu 22.04)
  FreeSurfer 8.1 · FSL · SimNIBS 4.5 · HippoUnfold · BigBrain · MRtrix3
  → surface meshes (cortex, cerebellum, hippocampus), tissue label volumes, atlas mappings
    │
    ▼
parrot_forward_model        (CGAL 6.1.1 / CMake / Python 3.12)
  CGAL mesher (C++) · pygeodesic · libigl · trimesh
  → volumetric mesh, dipole positions+normals, electrode positions
    │
    ▼
parrot_forward_solvers      (DUNEuro 2.10 / OpenMEEG 2.4)
  make_leadfield_duneuro.py or make_leadfield_openmeeg.py
  → leadfield matrix  (n_dipoles × n_electrodes)
```

Connectivity matrices in `connectivity/` drive TVB; they come in two resolutions (100-region and 1000-region atlases).

## Key Files to Know

| File | Role |
|------|------|
| `bin/run_reconstruction.sh` | BIDS-app entry point; most complex orchestration (~700 lines) |
| `bin/run_MRI_pipeline.sh` | Stage 1 driver; handles GPU flags and container invocation |
| `bin/run_forward_pipeline.sh` | Stage 2–3 driver |
| `containers/parrot_mri_reconstruction/scripts/run_cereb_pipeline.py` | Core MRI processing logic (~470 lines) |
| `containers/parrot_forward_model/place_dipoles.py` | Poisson-disk dipole sampling on surfaces |
| `containers/parrot_forward_solvers/make_leadfield_duneuro.py` | FEM leadfield via DUNEuro |
| `containers/parrot_fast_tvb/tvb_multicore.c` | AVX2-optimized TVB simulation engine |

## State Tracking

Pipeline stages detect already-completed runs via log files (not output file existence). If a stage appears to be skipped unexpectedly, check for stale log files in the subject's output directory.

## Connectivity Data

`connectivity/` holds pre-computed brain connectivity matrices at two resolutions. The `full_to_reduced_*.npy` / `reduced_to_full_*.npy` arrays map between the 1000-region and 100-region atlases. These are inputs to TVB, not outputs of the pipeline.
