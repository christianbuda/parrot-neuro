# Running Parrot on CINECA LEONARDO

LEONARDO has **no Docker** — only rootless **Apptainer/Singularity**. The pipeline
images run rootless (see the `feat/leonardo-apptainer-port` work), and
`bin/run_reconstruction.sh --runtime apptainer` drives them there. This directory
holds the cluster glue.

| File | Role |
|------|------|
| `prepull_sifs.sh` | Pull the 8 images into `<work>/parrot_sif` as `.sif` (run on a login/data-mover node). Re-pulls only images that changed on the registry (`FORCE=1` to re-pull all). |
| `check_leonardo.sh` | **Preflight** — run on a login node before `sbatch`; verifies runtime, `.sif` cache, BIDS+license+subject, repo, work area, account. Exits non-zero on any failure. |
| `pilot.sbatch` | One-subject end-to-end pilot on the Booster GPU partition, instrumented. |

> **Staying connected.** `sbatch` jobs run on the scheduler regardless of your SSH session —
> no `tmux` needed for the job. But the *foreground* login-node steps (`rsync`, `prepull_sifs.sh`)
> die if SSH drops, so run those inside `tmux`/`screen` (`tmux new -s parrot`, `Ctrl-b d` to
> detach, `tmux attach -t parrot` to return). `rsync -P` resumes partial transfers on re-run.

## One-time prerequisites

0. **Container runtime — confirmed.** On LEONARDO, **Singularity is a system command**
   (`/usr/bin/singularity`), *not* a module — it survives `module purge`, no `module load`
   needed. (There's no `apptainer` binary; our `--runtime apptainer` helper resolves to
   `singularity`. `module spider` isn't available on this Lmod, fwiw.) Verified 2026-06-25.
1. **Publish the rootless images.** From a machine with Docker:
   `./bin/build.sh --push` (pushes `christianbuda/parrot_*:latest`).
2. **Get the code on the cluster.** Clone this repo to `$HOME/parrot-neuro` (or edit the path in `pilot.sbatch`).
3. **Stage one subject's BIDS** to your work area (e.g. `/leonardo_work/<ACCT>/parrot/bids`), with
   `license.txt` at the dataset root, via the data-mover `data.leonardo.cineca.it` (rsync/rclone).
   The per-account work area is large (tens of TB on this allocation), so data **and** derivatives
   live there — the aggressive batched-`$SCRATCH`-purge workflow isn't needed at this scale. Use
   `/leonardo_scratch/fast/<ACCT>` only for hot per-job scratch if you want. Note `$WORK`/`$SCRATCH`
   env vars are ambiguous with multiple accounts → use explicit `/leonardo_work/<ACCT>` paths.
4. **Pre-pull the `.sif`s** on a login/data-mover node (compute nodes have no internet):
   ```bash
   bash hpc/leonardo/prepull_sifs.sh /leonardo_work/<ACCT>/parrot_sif
   ```

## Run the pilot

Edit the four `# <<EDIT>>` lines in `pilot.sbatch` (account, repo path, BIDS path, subject), then
**preflight before spending any GPU time** (pass the same values via env, or rely on the defaults):
```bash
ACCT=<ACCT> SUBJECT=<ID> bash hpc/leonardo/check_leonardo.sh   # must say PREFLIGHT PASSED
sbatch hpc/leonardo/pilot.sbatch
squeue --me            # PD = pending, R = running
```
It takes **one GPU** (`--gres=gpu:1`) and a 1/4-node CPU/RAM slice — Booster shares nodes and bills per-GPU, so this is the cheapest honest pilot. A **no-DWI** subject is fastest for a first run (skips the multi-hour QSIPrep/QSIRecon).

## Read the results — this is a measurement run

The pilot's job is to produce two numbers before we scale to a job array:
- **Total walltime vs the 24 h cap** (`boost_usr_prod`). Per-stage timings are in
  `<out>/logs/sub-<ID>/parrot-reconstruction_log.txt` and summarised at the end of the job's `.out`.
- **GPU-idle fraction** — written to `gpu_util-<jobid>.csv` and summarised in the `.out`. The recon
  container mixes GPU work (FastSurfer) with long CPU work (charm, surfaces); a high idle fraction is
  the argument for the **GPU/CPU split** (recon+DWI on Booster, mesher+solvers on DCGP).

**Decision gate:** if a subject fits comfortably in 24 h *and* GPU-idle is low → scale monolithic.
If it brushes the cap or idles the A100 for hours → build the split (a `--stages` selector +
a two-phase SLURM array with `--dependency=aftercorr`). Don't build the array before reading these.

### QoS options (for scaling — Booster `boost_usr_prod`)

| QoS | Walltime | Notes |
|-----|----------|-------|
| `normal` (default) | 24 h | standard production; the pilot uses this |
| `boost_qos_lprod` | **4 days** | long production, max 8 nodes / 32 GPUs per account — escape hatch if a subject brushes 24 h |
| `boost_qos_dbg` | 30 min | debug, **max 1 running-or-pending job**; too short for a full subject |
| `lrd_all_serial` (CPU partition) | 4 h | CPU-only, **budget-free**, ≤4 cores — ideal for **staging** and CPU-only forward stages |

`lrd_all_serial` being budget-free is the lever for Deliverable 2: run staging and the
mesher+solver (CPU) phase there for free, spend GPU-hours only on recon/DWI.

## Notes / gotchas
- **`signal: killed` while creating the squashfs/SIF** = the login node OOM-killed `mksquashfs`.
  Login-node `/tmp` is RAM-backed (tmpfs) and there's a per-user memory cap, so building a ~20 GB
  `.sif` there blows the limit. `prepull_sifs.sh` now redirects `APPTAINER_TMPDIR`/`CACHEDIR` to
  the disk-backed work FS, which fixes it. If the *biggest* image (`parrot_mri_reconstruction`,
  ~20 GB) still gets killed on the login node, run the same prepull on the **data-mover node**
  (`ssh <user>@data.leonardo.cineca.it`) — it has higher limits and is built for heavy I/O.
- Apptainer sets HOME via `--home` (it rejects `--env HOME`); the orchestrator handles this.
- Compute nodes lack internet → always `prepull_sifs.sh` first; the in-job auto-pull will fail otherwise.
- Don't max `--threads`: `place_dipoles` (and BLAS-heavy steps) oversubscribe badly. The pilot uses `--cpus-per-task`.
- Singularity is a **system command** (`/usr/bin/singularity`), *not* a module — no `module load` needed, survives `module purge`. `cuda/12.2` *is* a module (load it for `--nv`). `module spider` isn't available on this Lmod.
