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

## Installation

1.  **Clone the repository** (recursive is needed for submodules):
    ```bash
    git clone --recursive [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
    cd your-repo
    ```
2. **TODO**


## Usage

### 1. Processing Subjects (Bash Pipeline)
To run reconstruction and modeling on a set of subjects:

```bash
# Syntax: ./bin/run_pipeline.sh -s <subject_id> -d <subjects_dir> -t1 <t1 file>
./bin/run_pipeline.sh -s mni_nlin_asym_09b -d /path/to/subjects/dir -t1 t1.nii.gz
```

## Acknowledgments
https://github.com/ThomasYeoLab/CBIG
