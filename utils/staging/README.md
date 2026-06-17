# Dataset staging

Helpers that turn a **raw cohort dataset** into a flattened, cleaned BIDS dataset the
Parrot orchestrator can consume. This is a *pre-pipeline*, cohort-specific, one-off
step — distinct from the in-pipeline processing stages (e.g. MP2RAGE MPRAGEise lives
in the pipeline's `mp2rage_prep`, **not** here).

## Layout

| File | Role |
|------|------|
| `common.py` | Cohort-agnostic helpers: NIfTI header hygiene (`clean_voxel_size`, `copy_with_json`) + the `dataset_description.json` / `participants.tsv` writers. |
| `lemon.py` | LEMON-specific staging (session flattening, fmap `IntendedFor` fix, file map, per-subject overrides). |
| _future_ `hcp.py` | Same pattern for HCP. |

A new cohort = a new `<cohort>.py` reusing `common.py`; no launcher change needed.

## Running

Use the launcher — it runs the stager **inside the `parrot_mri_reconstruction`
image** (the host has no nibabel) and wires up the mounts:

```bash
./bin/stage.sh <cohort> <src_dir> <bids_out_dir> [subject ...]

# Example (LEMON):
./bin/stage.sh lemon \
    /srv/nfs-data/picard/christian/LEMON/.../MRI_MPILMBB_LEMON/MRI_Raw \
    /srv/nfs-data/sisko/christian/BIDS_LEMON \
    sub-010002
```

Omit the subject list to use the stager's built-in default.

## What staging does (and doesn't)

- **Does:** flatten sessions, copy only files Parrot consumes, snap the float32
  voxel-size header artifact (`1.0000009 -> 1.0`, geometry-preserving — FastSurfer's
  surf-stage conform rejects `vox_size > 1.0`), fix malformed fieldmap `IntendedFor`,
  and write `participants.tsv` with the orchestrator's positional override columns
  (`skip_t2_registration`, `no_neck`, `mp2rage`).
- **Doesn't:** touch image *intensities*. Raw MP2RAGE UNI is copied as-is and handled
  later by the pipeline's `mp2rage_prep` (gated by the `mp2rage` column).

## Per-subject overrides

Edit `SUBJECT_OVERRIDES` in the cohort script. The orchestrator parses the override
columns **positionally**, so keep `OVERRIDE_COLS` order in sync with what the
orchestrator expects.
