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
| `prewarm_hippunfold.sh` | **Populate the HippUnfold cache in place.** HippUnfold pulls atlases/templates from OSF at runtime, and **OSF is unreachable from LEONARDO compute nodes** — but *login* nodes have egress, so run it there to write straight into `<output_dir>/.hippunfold_cache`. Version-matched (reads URLs from the image config). Runtime + paths auto-detect (docker on the workstation, apptainer/singularity + `SIF_DIR`/paths from `config.local.sh` on a login node), so on a login node just run it with **no args**. |
| `prewarm_templateflow.sh` | **Populate the TemplateFlow cache in place** at `<output_dir>/.templateflow`. QSIPrep/QSIRecon fetch templates from S3 at runtime, unreachable from compute nodes. BUILD mode fetches via the QSIPrep image (runtime + paths auto-detect from `config.local.sh`, so **no args** on a login node); or `SRC=<a prior run's .templateflow>` to seed from an existing minimal cache. |
| `check_leonardo.sh` | **Preflight** — run on a login node before `sbatch`; verifies runtime, `.sif` cache, **HippUnfold + TemplateFlow caches**, BIDS+license+subject, repo, work area, account. Exits non-zero on any failure. Pass **`--fix`** to prewarm any missing cache in place (runs the `prewarm_*.sh` via singularity). |
| `pilot.sbatch` | One-subject end-to-end pilot on the Booster GPU partition, instrumented. |

**EEG+BOLD optimization stage** (`parrot_neuro.optimization`, per-subject JAX/TVB fit — a *separate* stage from reconstruction above; see its own section below):

| File | Role |
|------|------|
| `setup_optim_env.sh` | **One-time**: install `pixi` (if missing) + `pixi install` the optimization env (login node, needs internet). No container for this stage — a real Python/JAX env, self-contained CUDA via `jax[cuda12]` pip wheels. |
| `check_optim.sh` | **Preflight** — verifies the pixi env, and (via the real `parrot_neuro.Subject` code) that the target subject actually has the EEG/fMRI/leadfield derivatives this stage reads, plus output-dir writability and account. |
| `optim_cohort.sbatch` | Thin per-subject runner (mirrors `cohort.sbatch`) — resolves one subject, calls `examples/eeg_bold_fit_cli.py`. Never `sbatch` directly. |
| `submit_optim.sh` | Submitter with `smoke` / `pilot` / `run` / `list` subcommands — the single source of truth for GPU/CPU/walltime resources and the fit hyperparameters. |

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
   # runtime (apptainer) + cache paths + SIF_DIR are auto-detected from config.local.sh -> no args:
   bash hpc/leonardo/prewarm_hippunfold.sh
   bash hpc/leonardo/prewarm_templateflow.sh
   ```
   Or just let the preflight do it: **`bash hpc/leonardo/check_leonardo.sh --fix`** populates
   whichever cache is missing. (No egress on the login node? Seed TemplateFlow from a prior local
   run with `SRC=<…>/.templateflow`, or build on the workstation — where the runtime auto-detects
   as docker — and copy up.) The caches are shared across the whole cohort — do this once.
   `check_leonardo.sh` verifies both (and now rejects a TemplateFlow cache holding only the single
   artifact-setup canary file — it must be a full `tf.get`, or QSIPrep still hits S3 and dies).

## Run the pilot

With `config.local.sh` filled in (step 2), you don't edit `pilot.sbatch` at all —
**preflight, then submit via the wrapper** (which sources config and injects `--account`,
since a `#SBATCH` directive can't read a shell variable):
```bash
bash hpc/leonardo/check_leonardo.sh          # must say PREFLIGHT PASSED (add --fix to prewarm caches)
bash hpc/leonardo/submit_pilot.sh            # sources config, sbatch --account=$ACCT pilot.sbatch
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

## EEG+BOLD optimization stage

Fits the JAX-accelerated TVB (Jansen-Rit cortex / Wilson-Cowan subcortex) model to each
subject's own EEG + BOLD, per-node (`parrot_neuro.optimization`, driven by
`examples/eeg_bold_fit_cli.py`). This is downstream of, and **separate from**, the
reconstruction pipeline above: it reads a subject's already-computed leadfield + EEG +
fMRI derivatives, needs **no container**, and runs in a `pixi` environment instead
(the same one `pixi.toml` already builds for local dev — LEONARDO is just another
`linux-64` target). Assumes the subject is already reconstructed (i.e. `check_leonardo.sh`
/ the recon cohort has already produced its derivatives).

### One-time setup

```bash
# on a LOGIN node (needs internet: installs pixi + resolves conda-forge/PyPI):
bash hpc/leonardo/setup_optim_env.sh
```
This installs `pixi` to `$HOME/.pixi/bin` (if missing) and runs `pixi install` against
the repo's `pixi.toml`, producing a self-contained `.pixi/envs/default` — like the `.sif`
cache, built once with egress and reused offline on compute nodes afterward (no runtime
fetching, unlike HippUnfold/TemplateFlow). `jax[cuda12]` ships its own CUDA runtime as pip
wheels, so it only needs the compute node's NVIDIA driver — no `module load cuda` needed
for this stage.

### Preflight, then smoke test, then pilot, then scale

Same discipline as the reconstruction pilot: don't jump straight to a job array over an
unmeasured stage.

```bash
bash hpc/leonardo/check_optim.sh              # preflight: pixi env + subject derivatives + output dir
bash hpc/leonardo/submit_optim.sh smoke        # 1 subject, 2 epochs, debug QoS (~minutes) --
                                                #   "does the env + pipeline even run on a GPU node"
squeue --me                                    # watch it; check the .out for a clean finish, rc=0

bash hpc/leonardo/submit_optim.sh pilot        # 1 subject, FULL hyperparameters, timed + GPU-util logged
squeue --me
```
Read the pilot's `.out` for `optim finished in N min` and the GPU-idle summary (same
`gpu_util-<jobid>.csv` sampling as the reconstruction pilot). Set `OPTIM_TIME`/`OPTIM_MEM`/
`OPTIM_CPUS` in `config.local.sh` from what you observe (defaults are unmeasured
placeholders — `08:00:00` / `64G` / `8` cores), then scale:

```bash
bash hpc/leonardo/submit_optim.sh list         # sanity-check the array + resource matrix first
bash hpc/leonardo/submit_optim.sh run          # full job array over participants.tsv (same cohort list
                                                #   the reconstruction scripts use)
```
`run` accepts explicit subject labels as extra args for a small pilot or targeted retry
(`submit_optim.sh run 010002 010005`), mirroring `submit_cohort.sh run`. Each array task
takes one Booster GPU (`--gres=gpu:1`); `ARRAY_THROTTLE` (default `%40`) caps how many run
concurrently.

### Notes / gotchas (optimization stage)
- **Don't let this stage clobber SLURM's GPU binding.** `config.apply_jax_env()` used to
  hardcode `CUDA_VISIBLE_DEVICES` for a shared workstation (GPU index `3`); under `--gres`
  that variable is already scoped to the job's allocated device, so it now uses
  `setdefault` (only applies the workstation fallback when nothing has set it). No action
  needed on LEONARDO — this is just why it's safe.
- **JAX compilation cache off `$HOME`.** LEONARDO's `$HOME` is small and quota'd;
  `optim_cohort.sbatch` points `PARROT_JAX_CACHE_DIR` at
  `$WORKDIR/parrot/.jax_cache/sub-<ID>` (one subdir per subject — a job array hitting one
  shared cache concurrently risks lock contention).
- **Headless plotting.** `examples/eeg_bold_fit_cli.py` forces the `Agg` matplotlib backend
  before any `pyplot` import — compute nodes have no `DISPLAY`. Use `--skip-diagnostics`
  (the `smoke` subcommand always does) to skip the plotting section entirely for a faster
  sanity check.
- **Billing is per-GPU**, same as the reconstruction Booster chunks — one job = one A100,
  regardless of how many of its 32 cores you request.
- **GPU out-of-memory at atlas=1000 with the default 320s BOLD horizon?**
  Already fixed by default (`OPTIM_T1_WARMUP=30000` + `OPTIM_SOLVER_BLOCK_SIZE=565`
  in `optim_cohort.sbatch`/`submit_optim.sh`, plus a `del` in
  `eeg_bold_fit_cli.py`'s diagnostics section — see below) — this note explains
  why, in case you're tuning further or hit it at a different atlas/horizon.
  Validated on GPU 2026-07-29: atlas=1000 OOMs even on an **80G** card without
  the fixes. With them, measured peak was **~31GiB during training** and
  **~52GiB for a full run including the diagnostics/plotting section** (the
  default — `OPTIM_SKIP_DIAGNOSTICS=0`) — both comfortable under the A100's
  64G, though the full-run figure has less headroom (~12GiB) than the
  training-only one.

  There are **three** separate OOM sites in total, and each needed its own
  fix — none alone is sufficient:
  1. **The one-time BOLD warm-up solve** inside `build_simulators()` (seeds the
     network's initial state + a short delay-history buffer + the BOLD
     monitor's HRF-convolution tail). It's a plain forward call with no
     gradient, so `jax.checkpoint`-based blocking is a no-op there — yet it
     used to run for the *full* `t1_bold` (320s = 320k steps) just to throw
     away all but the last ~20s of it (none of its three consumers reads more
     than a short recent window). `t1_warmup` (`--t1-warmup`) gives this
     warm-up its own short, separate duration — it does **not** shorten the
     actual BOLD signal your FC/dFC loss sees (`t1_bold` is unchanged for the
     real training simulator); it only changes the exact initial state
     training starts from (a different, still-settled point, not a
     less-settled one — your own `bold_skip_trs=8` comment already implies the
     network settles in ~11.2s, well under the 30s default).
  2. **The real training step** (`bold_loss_fn`, wrapped in `jax.grad`): this
     one *is* differentiated, so `solver_block_size` (`--solver-block-size`,
     checkpoints the scan in blocks of `K` steps, `K ~ sqrt(n_steps)`) actually
     helps — `O(n_steps/K + K)` backward memory instead of `O(n_steps)`, for
     ~1.3-1.7x more compute, with the *exact* gradient (not an approximation).
     But the forward trajectory itself (~23GiB at atlas=1000/320s) is still a
     hard floor either way, since the BOLD monitor needs the whole thing.
  3. **The diagnostics/plotting section** (`eeg_bold_fit_cli.py`, skipped only
     by `--skip-diagnostics`) calls `ctx.simulators.simulator_bold(combined)`
     directly — forward-only again, same as #1, so `solver_block_size` doesn't
     help — and does it **twice** (fitted params, then again for `_init` to
     plot before/after). The raw ~23GiB trajectory from the first call used to
     stay referenced (only its small TR-downsampled extract was actually
     needed) while the second call's own ~23GiB trajectory was computed on top
     — two full trajectories resident at once. An explicit `del`, placed
     *before* the first call that actually forces materialization (getting
     this ordering right matters — a `del` placed after a forcing call is a
     no-op), releases the first trajectory before the second is computed.
  4. **The GPU allocator itself.** `config.apply_jax_env()` now forces
     `XLA_PYTHON_CLIENT_ALLOCATOR=platform` + `XLA_PYTHON_CLIENT_PREALLOCATE=false`
     together (both `setdefault`, so an explicit override still wins). This is
     not just cosmetic: verified empirically (2026-07-29) that JAX's *default*
     allocator (BFC — an arena that grows incrementally) reliably OOMs on the
     BOLD simulator's one-off ~23GiB allocation, reproduced on an
     otherwise-idle GPU with ~92GiB genuinely free — internal fragmentation,
     not an actual memory shortage. Setting `PREALLOCATE=false` *alone*
     (still BFC, just without the upfront grab) does **not** fix this; only
     `ALLOCATOR=platform` (direct cudaMalloc/cudaFree, no arena) does. Because
     this is now set inside `config.apply_jax_env()` itself, it applies
     however the script is invoked (SLURM, a bare `python examples/...`, a
     notebook) — not just under `optim_cohort.sbatch`.

  See `train.build_simulators`'s and `network.build_network`'s docstrings for
  the full accounting.

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
