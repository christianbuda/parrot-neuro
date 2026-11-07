# Parrot: Brain Simulation Framework

Parrot is a modular brain simulation tool that performs MRI reconstruction, electrical modeling, and simulation. It utilizes Docker containers for reproducible environments and provides a Python interface for easy interaction.

## Features

* **Reconstruction:** Automated pipeline using Freesurfer, SimNIBS (and others to add...).
* **Electrical Modeling:** Generates electrical properties for forward modelling.
* **Simulation:** The-Virtual-Brain based simulation engine.
* **Python API:** Simple interface to run simulations.

## Prerequisites

* **Docker:** Must be installed and running.
* **Linux Bash**

### Container versions

External dependencies are pinned by digest in `bin/images.sh` so upstream
`:latest` releases cannot silently break the pipeline. Parrot's own images are
pinned to one release version (`PARROT_VERSION`), which output manifests record. Docker and the LEONARDO SIF helpers share this manifest;
see [image compatibility and SIF cache migration](hpc/leonardo/README.md#image-versions).

## Installation

1.  **Clone the repository** (recursive is needed for submodules):
    ```bash
    git clone --recursive [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
    cd your-repo
    ```
2. **TODO**


## Usage

### Reading a subject in Python

```python
from parrot_neuro import Subject

s = Subject("/path/to/bids", "010002")  # or use PARROT_BIDS_ROOT
print(s.outputs)                        # registered output inventory
print(s.outputs.find("dipoles", spacing=2.0))
s.set_electrodes(["Cz", "Fz", "Pz"])
leadfield = s.load.leadfield("duneuroCGAL-2.0mm")
```

Paths require manifests under `derivatives/manifests/sub-<ID>/`. Reconstruction
and EEG/fMRI staging write them at successful step completion, using the actual
producer image reference. Existing datasets need manifest population separately;
skipped historical steps are not assigned new provenance. Rebuild the QC image
(`./bin/build.sh qc`) before running the updated reconstruction hooks.

`s.path` returns original files. `s.load.leadfield` selects model electrodes in
caller order and average-references across those rows. Allowed/default names come
from the reconstruction-selected JSON; original solver rows follow the full CSV
montage. `s.electrode_names` and `s.load.electrodes()` describe the current model
selection. `set_electrodes(None)` restores its default. Recorded EEG and its
channel metadata are independent; previously returned arrays do not change.

Anatomy, surfaces, dipoles, metadata and transformed leadfields are lazy and cached.
Cached objects are mutable and shared: copy before modifying them. EEG/BOLD and
large distances are not cached by default. Volume `use_cache=False` bypasses
lookup and insertion; `clear_cache()` releases retained objects. The constructor
has no global cache switch. `has_*` checks physical presence only, not manifest
registration or QC. Existing QC warn/fail stages are summarized at initialization.

For fMRI alignment, pass the same `task` to weights, distances, connectivity
labels, BOLD, and dipole-node mappings. Alignment selects usable nodes; it is not
spatial registration. `normalized=True` on weights means inverse-node-volume
weights, not max-normalization. See `examples/subject_usage.py` for a fuller tour.

`bin/stage.sh` supplies staging provenance and the shared QC helper. Standalone
EEG/fMRI container launches must likewise set `PARROT_PRODUCER_TAG` to the image
reference actually used and expose `containers/parrot_qc` on `PYTHONPATH` (mount
its `qc/` package when running in a container). Only newly completed outputs are
registered. The manifest writer still knows producer filenames; changing a
producer layout requires updating its selections and the contract tests.

### 1. Processing Subjects (Bash Pipeline)
To run reconstruction and modeling on a set of subjects:

```bash
# Syntax: ./bin/run_pipeline.sh -s <subject_id> -d <subjects_dir> -t1 <t1 file>
./bin/run_pipeline.sh -s mni_nlin_asym_09b -d /path/to/subjects/dir -t1 t1.nii.gz
```

## Acknowledgments
https://github.com/ThomasYeoLab/CBIG
