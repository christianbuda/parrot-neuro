# Running Parrot on CINECA LEONARDO

LEONARDO has **no Docker** — only rootless **Apptainer/Singularity**. The pipeline
images run rootless (see the `feat/leonardo-apptainer-port` work), and
`bin/run_reconstruction.sh --runtime apptainer` drives them there. This directory
holds the cluster glue.

| File | Role |
|------|------|
| `prepull_sifs.sh` | Pull the 8 images into `<work>/parrot_sif` as `.sif` (run on a login/data-mover node). Re-pulls only images that changed on the registry (`FORCE=1` to re-pull all). |
| `build_sif_fallback.sh` + `build_sif.sbatch` | **Two-phase build** for when `prepull_sifs.sh` OOM-kills on the login node (multi-GB images). Phase A (login) *downloads* each image to a single archive via skopeo/crane — no extraction; Phase B (`lrd_all_serial` job) does the extract+squashfs into `.sif` with real memory. |
| `build_sif_local.sh` | **Build the `.sif` on your workstation** (real RAM → never OOMs) and rsync them up. The deterministic alternative to the two-phase gamble; recommended for the cohort run. Needs local `apptainer` + `sudo` (see note below). |
| `prewarm_hippunfold.sh` | **Populate the HippUnfold cache in place.** HippUnfold pulls atlases/templates from OSF at runtime, and **OSF is unreachable from LEONARDO compute nodes** — but *login* nodes have egress, so run it there (`RUNTIME=apptainer SIF_DIR=…`) to write straight into `<output_dir>/.hippunfold_cache`. Version-matched (reads URLs from the image config). Runs under docker (workstation) or singularity (login node). |
| `prewarm_templateflow.sh` | **Populate the TemplateFlow cache in place** at `<output_dir>/.templateflow`. QSIPrep/QSIRecon fetch templates from S3 at runtime, unreachable from compute nodes. BUILD mode fetches via the QSIPrep image (login node: `RUNTIME=apptainer SIF_DIR=…`); or `SRC=<a prior run's .templateflow>` to seed from an existing minimal cache. |
| `check_leonardo.sh` | **Preflight** — run on a login node before `sbatch`; verifies runtime, `.sif` cache, **HippUnfold + TemplateFlow caches**, BIDS+license+subject, repo, work area, account. Exits non-zero on any failure. Pass **`--fix`** to prewarm any missing cache in place (runs the `prewarm_*.sh` via singularity). |
| `pilot.sbatch` | One-subject end-to-end pilot on the Booster GPU partition, instrumented. |

> **Staying connected.** `sbatch` jobs run on the scheduler regardless of your SSH session —
> no `tmux` needed for the job. But the *foreground* login-node steps (`rsync`, `prepull_sifs.sh`)
> die if SSH drops, so run those inside `tmux`/`screen` (`tmux new -s parrot`, `Ctrl-b d` to
> detach, `tmux attach -t parrot` to return). `rsync -P` resumes partial transfers on re-run.

## Getting the `.sif` images — three routes

The compute nodes have **no internet**, so the `.sif` cache must be populated ahead of time.
There are three ways to do it; they produce the same cache, so pick by what fails on you.

| # | Route | Script(s) | When to use | Cost / failure mode |
|---|-------|-----------|-------------|---------------------|
| 1 | **Direct pull on login node** | `prepull_sifs.sh` | First thing to try — one command, digest-aware (re-pulls only changed images) | Login node OOM-kills `mksquashfs` (or even the *extract*) on the multi-GB images → `signal: killed` |
| 2 | **Two-phase (download on login, squashfs in a job)** | `build_sif_fallback.sh` (Phase A) + `build_sif.sbatch` (Phase B) | When #1 OOMs and you have no local box / don't want to transfer | Phase B may still OOM `mksquashfs` inside the serial-QoS 30 G cap on the biggest images → fall back to `SANDBOX=1 sbatch …` (extract-only, no squashfs; slower runtime on Lustre) |
| 3 | **Build locally, rsync up** | `build_sif_local.sh` | **Recommended for the cohort run** — deterministic, a workstation has real RAM so it never OOMs, and you get proper single-file `.sif` | Needs local Docker + `apptainer` + one-time `sudo`; costs a ~tens-of-GB upload |

**Why the memory pain (#1/#2) exists at all:** building a `.sif` *extracts* the image into a
full root-owned filesystem tree (root:root files, setuid bits) and then packs it with
`mksquashfs`. Both steps size their buffers from the node's *total physical RAM* (~512 G on a
Booster node), not from your cgroup, so on a memory-capped login node — or the 30 G serial-QoS
slice — they get OOM/arbiter-killed. A workstation (route 3) has the RAM, so it just works.

**Why route 3 needs `sudo` but LEONARDO's build (routes 1/2) doesn't:** the chown/setuid-preserving
extraction needs privilege from *one* of three sources — real root (`sudo`), a trusted
**setuid-root** helper, or **unprivileged user namespaces**. LEONARDO (RHEL8) provides the last
two (userns is allowed; SingularityPRO is installed setuid-root by the admins), so no `sudo`
there. An Ubuntu 23.10+/24.04 workstation blocks unprivileged userns
(`kernel.apparmor_restrict_unprivileged_userns=1`) and ships plain `apptainer` (no suid helper),
so only real root is left → `sudo apptainer build`. Building as root does **not** make the `.sif`
need root at runtime; it still runs rootless on LEONARDO. (To build locally *without* sudo you'd
re-enable one of the other two: `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0`, or
`apt install apptainer-suid` — each a one-time sudo.)

Whichever route you use, verify the result with `check_leonardo.sh` (it accepts `<name>.sif` or a
`<name>/` sandbox dir).

## One-time prerequisites

0. **Container runtime — confirmed.** On LEONARDO, **Singularity is a system command**
   (`/usr/bin/singularity`), *not* a module — it survives `module purge`, no `module load`
   needed. (There's no `apptainer` binary; our `--runtime apptainer` helper resolves to
   `singularity`. `module spider` isn't available on this Lmod, fwiw.) Verified 2026-06-25.
1. **Publish the rootless images.** From a machine with Docker:
   `./bin/build.sh --push` (pushes `christianbuda/parrot_*:latest`).
2. **Get the code on the cluster.** Clone this repo to `$HOME/parrot-neuro` (or edit the path in `pilot.sbatch`).
   Then set your account + paths **once** in a gitignored config, so you never hand-edit the scripts:
   ```bash
   cp hpc/leonardo/config.local.sh.example hpc/leonardo/config.local.sh   # then edit ACCT + paths
   ```
   `check_leonardo.sh`, `pilot.sbatch`, and both `prewarm_*.sh` source it automatically (env vars still
   override). It's found whether you run a script directly or via `sbatch` (resolved from `$SLURM_SUBMIT_DIR`
   / `$HOME/parrot-neuro`), or point anywhere with `PARROT_CONFIG=/path/to/config.local.sh`.
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
5. **Pre-warm the runtime-fetch caches (HippUnfold + TemplateFlow).** Both HippUnfold (OSF,
   `files.ca-1.osf.io`) and QSIPrep/QSIRecon (TemplateFlow, S3) download assets at runtime, and
   **both hosts are unreachable from LEONARDO compute nodes** (`Network is unreachable`). But
   *login* nodes have egress, so populate the caches **in place** on a login node — no rsync:
   ```bash
   # on a LOGIN node (has egress + singularity + the .sif cache):
   OUT=/leonardo_work/<ACCT>/parrot/bids/derivatives
   SIF=/leonardo_work/<ACCT>/parrot/parrot_sif
   RUNTIME=apptainer SIF_DIR=$SIF bash hpc/leonardo/prewarm_hippunfold.sh   "$OUT/.hippunfold_cache"
   RUNTIME=apptainer SIF_DIR=$SIF bash hpc/leonardo/prewarm_templateflow.sh "$OUT/.templateflow"
   ```
   Or just let the preflight do it: **`bash hpc/leonardo/check_leonardo.sh --fix`** populates
   whichever cache is missing. (No egress on the login node? Seed TemplateFlow from a prior local
   run with `SRC=<…>/.templateflow`, or build on the workstation with `RUNTIME=docker` and copy up.)
   The caches are shared across the whole cohort — do this once. `check_leonardo.sh` verifies both.

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
- **`signal: killed` while building a `.sif`** = the login node OOM/arbiter-killed it. Login
  nodes have a small per-user memory cap and a RAM-backed `/tmp`, so building a multi-GB `.sif`
  there dies — at the `mksquashfs` step, or even at *extract* for the ~20 GB images. Redirecting
  tmp/cache to the disk FS (which `prepull_sifs.sh` does) is not always enough, because the
  extract/squashfs itself needs the memory. With no interactive data-mover login, use the
  **two-phase build**, which never extracts on the login node:
  ```bash
  # Phase A (login node, internet, memory-light): DOWNLOAD each image to an archive (no extract)
  bash hpc/leonardo/build_sif_fallback.sh /leonardo_work/<ACCT>/parrot_sif
  # Phase B (budget-free serial job, real --mem): extract+squashfs archive -> .sif (no internet)
  sbatch hpc/leonardo/build_sif.sbatch /leonardo_work/<ACCT>/parrot_sif   # edit the account line
  bash hpc/leonardo/check_leonardo.sh                                     # confirm all 8 .sif
  rm -rf /leonardo_work/<ACCT>/parrot_sif/.staging/*.tar                  # reclaim space after
  ```
  Phase A uses `skopeo`/`crane` (auto-fetches a static `crane` if neither is installed). Phase B
  is idempotent — if it hits the 4 h wall, just re-submit and it resumes.
- **The deterministic alternative: build `.sif` locally.** The two-phase route only exists
  because LEONARDO's login/serial memory is too small for `mksquashfs` on the big images. A
  workstation has real RAM, so `build_sif_local.sh` just builds proper single-file `.sif`
  and rsyncs them up — no OOM, no sandbox. **Recommended for the cohort run.**
  ```bash
  # local box with Docker + apptainer (sudo add-apt-repository -y ppa:apptainer/ppa; apt install apptainer)
  bash hpc/leonardo/build_sif_local.sh                       # all 8 -> ./parrot_sif_local
  # or push straight to the login node (host/path kept out of git):
  DEST=<USER>@login.leonardo.cineca.it:/leonardo_work/<ACCT>/parrot/parrot_sif/ \
      bash hpc/leonardo/build_sif_local.sh
  ```
  It builds as **root** (`sudo apptainer build`): extracting an image into a root-owned tree
  with setuid bits needs real root or unprivileged user namespaces, and Ubuntu 23.10+/24.04
  block the latter (`kernel.apparmor_restrict_unprivileged_userns=1`). Building as root does
  **not** make the `.sif` need root at runtime — it still runs rootless on LEONARDO. Prefers
  your local Docker image when present (fast); falls back to Docker Hub otherwise.
- Apptainer sets HOME via `--home` (it rejects `--env HOME`); the orchestrator handles this.
- Compute nodes lack internet → always `prepull_sifs.sh` first; the in-job auto-pull will fail otherwise.
- Don't max `--threads`: `place_dipoles` (and BLAS-heavy steps) oversubscribe badly. The pilot uses `--cpus-per-task`.
- Singularity is a **system command** (`/usr/bin/singularity`), *not* a module — no `module load` needed, survives `module purge`. `cuda/12.2` *is* a module (load it for `--nv`). `module spider` isn't available on this Lmod.
