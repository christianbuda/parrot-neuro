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
- **BOLD loss is now always a weighted combination of static FC + dFC/FCD**
  (`OPTIM_BOLD_FC_WEIGHT`/`OPTIM_BOLD_DFC_WEIGHT`, default `0.5`/`0.5`), replacing the old
  exclusive `OPTIM_BOLD_LOSS=fc|dfc` selector — both terms come from the SAME simulated
  trajectory (see `train.make_bold_loss_fn`), so combining them doesn't double the cost of
  the expensive BOLD forward pass. Set either weight to `0` to recover a single-mode fit.
  This also changed the output directory naming: new runs land under
  `<subject>_<optimize>/` (no more `_fc`/`_dfc` suffix) — old saved directories/npz files
  are untouched and still fully readable via `postfit_diagnostics_cli.py`.
- **Two new optional, off-by-default loss terms**: `OPTIM_BOLD_PSD_WEIGHT` (a Welch-PSD
  spectral-shape term for BOLD, restricted to the 0.01-0.1Hz bandpass — `fc_vector`'s
  time-averaged correlation has no sensitivity at all to each signal's own temporal/
  spectral shape) and `OPTIM_GAMMA_WEIGHT` (a log(PSD) MSE term for EEG over 15-40Hz,
  alongside the existing 1-15Hz normalized-linear term). Both default to `0` (off).
  `OPTIM_LEARNING_RATE_BOLD` (default empty = reuse `OPTIM_LEARNING_RATE`) lets the BOLD
  step use a different rate than EEG now that they have separate optimizer state.
- **Early stopping is off by default.** Every array task runs the full `OPTIM_NUM_EPOCHS`
  unless you set `OPTIM_EARLY_STOP_PATIENCE` (in `config.local.sh` or as a call-time env
  var) — the fit then stops once every actively-optimized loss's trend over the last
  `OPTIM_EARLY_STOP_PATIENCE` overlapping `OPTIM_EARLY_STOP_WINDOW`-sized windows has
  stayed flat or increasing (see `train.is_loss_stalled`). Cheaper cohort runs once you've
  picked a patience that looks safe on a pilot's loss curves — check the `.out` for the
  `Early stopping at epoch N` line and how many entries `loss_history_*.npy` actually has.
- **Alternating "both" fits now keep separate optimizer state per loss** (EEG PSD vs.
  BOLD FC) instead of one shared Adam state — fixes a real bug where the BOLD loss barely
  moved under `optimize=both` even though a BOLD-only fit converged fine at the same epoch
  count. No config change needed; this is automatic once the cluster's clone is on a
  commit that includes it.
- **`bold_timeseries.png`/`bold_learning.png` used to bandpass-filter simulated BOLD
  before slicing off the burn-in instead of after**, which smeared the unsettled onset
  transient's ringing across the whole plotted trace (looked like spurious high-frequency
  content). Fixed to filter after slicing, matching `fc_comparison.png`'s already-correct
  order — the actual fitted loss was never affected, only these two diagnostic plots.
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
- **GPU out-of-memory at atlas=1000 with a long BOLD horizon?**
  `t1_bold` defaults to 700s (500 TRs at `tr_ms=1400` — up from an original
  320s default; the FC/dFC loss wants as long a BOLD signal as is affordable).
  A longer horizon needs correspondingly more fixes than the historical
  320s-tuned defaults below provided — see "Streaming BOLD monitor" right
  after this note for the current (2026-08) fix, which changes the memory
  picture qualitatively, not just by how much headroom the old fixes bought.
  The **four** OOM sites below (validated on GPU 2026-07-29, at the *old*
  320s default) are kept for historical/debugging context — #1 and #4 are
  still exactly as described; #2 now ALSO streams the BOLD monitor's HRF
  convolution through the same block scan (bigger win than blocking alone);
  #3 (diagnostics double-materializing the raw trajectory) no longer applies
  at all, because nothing downstream of `simulator_bold(...)` materializes
  the raw per-ms trajectory anymore, training or diagnostics.
  1. **The one-time BOLD warm-up solve** inside `build_simulators()` (seeds the
     network's initial state + a short delay-history buffer + the BOLD
     monitor's HRF-convolution tail). It's a plain forward call with no
     gradient, so `jax.checkpoint`-based blocking is a no-op there — yet it
     used to run for the *full* `t1_bold` just to throw away all but the last
     ~20s of it (none of its three consumers reads more than a short recent
     window). `t1_warmup` (`--t1-warmup`) gives this warm-up its own short,
     separate duration — it does **not** shorten the actual BOLD signal your
     FC/dFC loss sees (`t1_bold` is unchanged for the real training
     simulator); it only changes the exact initial state training starts
     from (a different, still-settled point, not a less-settled one — your
     own `bold_skip_trs=8` comment already implies the network settles in
     ~11.2s, well under the 30s default).
  2. **The real training step** (`bold_loss_fn`, wrapped in `jax.grad`): this
     one *is* differentiated, so `solver_block_size` (`--solver-block-size`,
     checkpoints the scan in blocks of `K` steps) helps — `O(n_steps/K + K)`
     backward memory instead of `O(n_steps)`, for ~1.3-1.7x more compute,
     with the *exact* gradient (not an approximation). As of the streaming
     BOLD monitor (below), this `K` is now the *only* memory knob for BOLD —
     see that section for why it must be an exact multiple of 1400, not just
     `~sqrt(n_steps)`.
  3. ~~The diagnostics/plotting section calling `simulator_bold` twice~~ —
     moot now: neither call materializes the raw trajectory anymore (see
     "Streaming BOLD monitor"). Kept here only so old commit history /
     issue reports referencing this fix still make sense.
  4. **The GPU allocator itself.** `config.apply_jax_env()` forces
     `XLA_PYTHON_CLIENT_ALLOCATOR=platform` + `XLA_PYTHON_CLIENT_PREALLOCATE=false`
     together (both `setdefault`, so an explicit override still wins). This is
     not just cosmetic: verified empirically (2026-07-29) that JAX's *default*
     allocator (BFC — an arena that grows incrementally) reliably OOMs on a
     large one-off allocation, reproduced on an otherwise-idle GPU with
     ~92GiB genuinely free — internal fragmentation, not an actual memory
     shortage. Setting `PREALLOCATE=false` *alone* (still BFC, just without
     the upfront grab) does **not** fix this; only `ALLOCATOR=platform`
     (direct cudaMalloc/cudaFree, no arena) does. Because this is set inside
     `config.apply_jax_env()` itself, it applies however the script is
     invoked (SLURM, a bare `python examples/...`, a notebook).

  See `train.build_simulators`'s and `network.build_network`'s docstrings for
  the full accounting.

- **Streaming BOLD monitor (2026-08) — the current fix for long `t1_bold`.**
  The BOLD "monitor" (`HRFBold`, from `tvboptim`) convolves the raw simulated
  trajectory with the HRF kernel — historically *after* the full solve, on
  the *entire* stacked `(n_steps, n_voi, n_nodes)` trajectory (a batched
  `jax.scipy.signal.fftconvolve`). That's a hard memory floor independent of
  `solver_block_size` (which only checkpoints the solver's *backward* pass,
  not this separate post-hoc step) — and it scales directly with `t1_bold`,
  so simply raising `t1_bold` (e.g. 320s → 900s) can make the cuFFT scratch
  allocation itself fail outright (`RET_CHECK failure ... Failed to create
  cuFFT batched plan`), not just OOM more gracefully.

  `train.build_simulators` now passes `reduce=streaming_hrf_bold(bold_monitor,
  dt)` into the BOLD `prepare()` call (`tvboptim`'s own purpose-built fix for
  this): the HRF convolution runs block-by-block, folded into the same
  `jax.checkpoint`'d scan `solver_block_size` already drives, and
  `simulator_bold(combined)` returns the final small `(n_bold, n_voi,
  n_nodes)` BOLD buffer directly — the full raw trajectory is never
  materialized at all, for training *or* diagnostics (both call the same
  `simulator_bold`). Verified numerically equivalent to the old post-hoc
  `bold_monitor(sol)` call (max abs diff ~1e-14 on synthetic data, both with
  and without warm-start history) — this is a memory-schedule change, not an
  approximation.

  Two things this required changing, both load-bearing:
  - **`solver_block_size` must be an exact multiple of the BOLD period in raw
    steps** (`tr_ms / dt` — 1400 for the defaults), not just close to
    `sqrt(n_steps)` — `streaming_hrf_bold`'s per-block update asserts this.
    The default changed from `565` to `1400` (one TR per block, the smallest
    valid choice) across `optim_cohort.sbatch`/`submit_optim.sh`,
    `sweep_train.sbatch`, and both CLI scripts' `--solver-block-size`.
  - **The neural-activity downsampling before HRF convolution switched from
    `TemporalAverage` (mean over each 4ms window) to `SubSampling` (pick 1
    sample per 4ms window)** — `streaming_hrf_bold` requires a
    uniform-integer-stride downsampler; its per-block update always does a
    hard-coded "take every Nth sample" slice regardless of what
    `monitor.downsample` actually is, so anything else would silently desync
    the streaming path from the (now theoretical) post-hoc one. This is a
    small but real change to what the BOLD loss computes, not merely a
    performance fix — `train.build_simulators` builds the monitor with an
    explicit `downsample=SubSampling(...)` now, not `HRFBold`'s own default.

## Hyperparameter sweep (wandb)

A W&B **Bayesian sweep** over the optimization stage's learning rates, BOLD
loss-term weights, and dFC window size — each trial fits a **fixed list of
subjects** (`SWEEP_SUBJECTS`) with one sampled hyperparameter set, in one
wandb run, logging per-subject curves/plots as well as an
`aggregate/combined_loss` objective (mean final EEG+BOLD loss across
subjects) the sweep minimizes. Driven by `examples/eeg_bold_fit_sweep.py`
(a twin of `eeg_bold_fit_cli.py` that loops over subjects instead of taking
one) plus `hpc/leonardo/sweep_eeg_bold.yaml`, `sweep_dispatch.sh`,
`sweep_train.sbatch`, `submit_sweep.sh`.

**Why this isn't just "`submit_optim.sh` but with a sweep flag":** `wandb
agent` needs live access to `api.wandb.ai`, both to fetch each trial's
Bayesian-suggested config and to stream `wandb.log()` calls — but this
README already established that **LEONARDO's compute nodes have no internet**
(only login nodes do). So the agent runs on the **login node**, and hands
each trial off to a compute node via `sbatch --wait`, training fully
offline (`WANDB_MODE=offline`) there, and syncing the result back once the
job returns and the login node has egress again:

```
login node (has egress)                    compute node (Booster GPU, NO egress)
wandb agent <SWEEP_ID>  (x8, one per worker)
  │ polls api.wandb.ai for the next Bayes-suggested config
  ▼
sweep_dispatch.sh --learning_rate=... ...
  │ WANDB_RUN_ID/WANDB_SWEEP_ID/WANDB_API_KEY already in env (wandb agent sets them)
  │ sbatch --wait --export=ALL,WANDB_MODE=offline,...  ──▶  sweep_train.sbatch
  │                                                           pixi run python -u
  │                                                             examples/eeg_bold_fit_sweep.py
  │                                                           loops over $SWEEP_SUBJECTS,
  │                                                           wandb.init(mode="offline", id=$WANDB_RUN_ID)
  │  ◀── job exit code ──────────────────────────────────── writes offline run dir
  ▼
wandb sync --id "$WANDB_RUN_ID" <offline run dir>
exit  →  agent sees the trial finished, polls for the next one
```

### One-time setup

Add to `config.local.sh` (see `config.local.sh.example`'s "W&B hyperparameter
sweep" block): `WANDB_API_KEY` (from https://wandb.ai/authorize) and
`SWEEP_SUBJECTS` (the fixed subject list every trial fits, e.g.
`010002,010003,010004,010005,010006`). Everything else has a default.

### Smoke test first — mandatory

This chain has several hops (login-node agent → `sbatch --wait` → offline
training → `wandb sync`); validate all of them with one cheap trial before
committing real GPU-hours:

```bash
bash hpc/leonardo/submit_sweep.sh create      # registers the sweep, saves its ID
bash hpc/leonardo/submit_sweep.sh smoke       # ONE trial, 1 subject, 2 epochs, foreground
```
Confirm: the compute job appears in `squeue --me`, finishes with `rc=0`, and
the trial shows up as a **finished** run on the wandb dashboard (not stuck
"crashed" from a failed sync). Only then scale up:

```bash
bash hpc/leonardo/submit_sweep.sh start 8 5   # 8 background agents x 5 runs each = 40 trials
bash hpc/leonardo/submit_sweep.sh status      # squeue + how many agents are still alive
bash hpc/leonardo/submit_sweep.sh stop        # kill the background agents
```

### Parallel (whole-node) mode

Sequential mode fits `SWEEP_SUBJECTS` one GPU/subject at a time — safe, but
capped by `boost_qos_lprod`'s 32-GPU-per-Project-Account ceiling (shared with
everyone else on the account) and needing its 4-day QoS just to fit the
walltime. `eeg_bold_fit_sweep.py --gpus 0,1,2,3` (set via `SWEEP_GPUS` in
`config.local.sh`) fits that many subjects **simultaneously** instead, as
separate `eeg_bold_fit_cli.py` subprocesses, one pinned per GPU — see the
module's own docstring for the mechanism (round-chunking, per-GPU JAX cache
isolation, wandb result replay since a subprocess can't log live).

**Measured on this project (2026-08):** one subject's full 300-epoch fit
takes **~6.6h** (`submit_optim.sh pilot`, subject 010002). That changes the
whole cost/QoS picture:

| Mode | Trial time (5 subjects) | Core-hours/trial | QoS |
|---|---|---|---|
| Sequential (1 GPU) | ~33.2h | `33.2h × 8 ≈ 266` | `boost_qos_lprod` (4-day; 33.2h exceeds `normal`'s 24h) |
| Parallel, 4 subjects/4 GPUs (1 round, **zero idle GPU-time**) | ~6.6h | `6.6h × 32 ≈ 212` (same total as sequential — no waste) | `normal` (24h, no documented account-wide GPU cap) |
| Parallel, 5 subjects/4 GPUs (2 rounds — 3 GPUs idle in round 2) | ~13.3h | `13.3h × 32 ≈ 425` (~60% more than sequential) | `normal`, comfortably |

CINECA bills `T(hours) × N(nodes) × R(max reserved fraction) × C(32 cores/node)`
— not for actual FLOPs, for what you *reserve* the whole job. Requesting all 4
GPUs pins `R=1.0` (the whole node) for the full trial duration; any round
where fewer than 4 of those 4 reserved GPUs are actually working still bills
as if all 4 were. **Match `SWEEP_SUBJECTS`' count to `len(SWEEP_GPUS)`** (e.g.
exactly 4 subjects for a 4-GPU node) to get the 2.5-5x wall-clock speedup
*and* escape the `lprod` ceiling with **zero** extra core-hour cost — a
remainder (5 subjects on 4 GPUs) still works correctly, just pays for the
idle GPUs during the odd last round.

`sweep_dispatch.sh` derives `--gres=gpu:<count>` from `SWEEP_GPUS`, defaults
`SWEEP_QOS` to `normal` automatically once it's set (still overridable), caps
`SWEEP_TIME`'s default at `20:00:00` (under `normal`'s 24h hard limit — widen
it, still `<24h`, if your subject count needs more rounds), and scales
`SWEEP_CPUS`/`SWEEP_MEM` by GPU count (8 cores/64G per GPU, matching the
per-subject sequential defaults — free from a billing perspective once `R`
is already 1.0 from the GPU request alone). Test locally first
(`pixi run python examples/eeg_bold_fit_sweep.py --gpus 0,1,2,3 ...`, no
SLURM needed) before trusting it on Leonardo — a `submit_sweep.sh smoke` with
`SWEEP_GPUS` set is still the right first Leonardo validation step.

### Notes / gotchas (sweep)
- **A sequential trial fits `SWEEP_SUBJECTS` one at a time at full epoch
  count** — several times a single-subject `submit_optim.sh` fit's walltime.
  `sweep_dispatch.sh` defaults its QoS to `boost_qos_lprod` (4-day wall), not
  `normal` (24h), for exactly this reason when `SWEEP_GPUS` is unset — still,
  measure with `smoke`/a short manual run before trusting `SWEEP_TIME`'s
  default. See "Parallel (whole-node) mode" above for the alternative.
- **`start`'s background `wandb agent` processes live on the login node for
  as long as trials keep dispatching** (hours to days) — run `start` inside
  `tmux`/`screen`, not a plain interactive shell that dies on logout. Leonardo
  has multiple login nodes behind one round-robin alias (`login.leonardo.
  cineca.it`), so a *later* SSH connection can land you on a different node
  than the one your session is running on — reattach to the specific node via
  its `-ext` hostname (e.g. `ssh <user>@login07-ext.leonardo.cineca.it`; only
  `login01/02/05/07` are documented with `-ext` names), not the round-robin
  alias, or `tmux ls` will correctly-but-confusingly report no sessions.
- **A login-node session dying mid-sweep can strand finished trials unsynced.**
  `sweep_dispatch.sh` only runs `wandb sync` *after* `sbatch --wait` returns —
  if the login-node process gets killed while still waiting (session loss,
  not something you did wrong), the compute job keeps running to completion
  independently (SLURM jobs don't depend on the submitting process), but
  nothing is left alive to sync its result afterward. The data isn't lost,
  just sitting unsynced in `$WORKDIR/parrot/wandb_offline/`. Recover with:
  `bash hpc/leonardo/sync_orphaned_runs.sh` — safe to run over everything,
  not just the orphaned ones (`wandb sync` is idempotent).
- **Bayesian search only sees a trial once it's synced.** If a trial's
  compute job OOMs or crashes, `sweep_dispatch.sh` still attempts a sync (of
  whatever got logged before the crash) and propagates the failing exit
  code — check `sweep_logs/agent-<i>.log` and the wandb dashboard if a trial
  shows up as failed/incomplete.
- **The 7 swept fields** (`learning_rate`, `learning_rate_bold`,
  `bold_fc_weight`, `bold_dfc_weight`, `bold_psd_weight`, `dfc_window_trs`,
  `dfc_step_trs`) are defined in `sweep_eeg_bold.yaml`; everything else
  (`atlas`, `num_epochs`, `optimize`, etc.) is fixed across the whole sweep
  via `SWEEP_*` env vars, same as `OPTIM_*` for `submit_optim.sh`.
- **`start N ...` launching far fewer than N agents that actually run** — two
  independent causes, both a login-node `RLIMIT_NPROC` squeeze:
  1. If `wandb` isn't directly on `PATH` (the normal case, needing `pixi
     run`), wrapping every one of N nearly-simultaneous agent launches in its
     own `pixi run` re-initializes `pixi`'s internal thread pool (`rayon`) N
     times within the same second or two, which can make `pixi` itself panic
     (`failed to initialize global rayon pool: ... Resource temporarily
     unavailable`) *before* `wandb agent` ever starts. `submit_sweep.sh`
     activates the pixi env **once** (`pixi shell-hook`, evaluated into the
     current shell) instead of per-agent, which avoids this entirely.
  2. **A different source of the same ceiling, confirmed 2026-08-30 at
     N=24**: each `wandb agent` process spins up its own internal asyncio
     "service" thread immediately on startup, independent of `pixi`. Even
     with (1) fixed, a burst of N of these within a few seconds can still
     exhaust `RLIMIT_NPROC` — a few agents die with `RuntimeError: can't
     start new thread` or a bare `SIGSEGV`, and never produce a single run.
     `submit_sweep.sh` staggers launches by `SWEEP_AGENT_STAGGER` seconds
     each (default 2, was a hardcoded 0.2 which wasn't enough) — ~4.3 min
     total launch time for N=128 at the default, negligible against
     hours-long trials. If you still lose agents at launch, widen it (e.g.
     `SWEEP_AGENT_STAGGER=60 ./submit_sweep.sh start 24 5`) or reduce `N`.
     Widening this further than strictly needed for the launch burst is
     also a legitimate way to reduce *sustained* login-node load: agents
     launched close together tend to run similar-length trials and so
     finish (and hit the `wandb sync` OpenBLAS pressure below) close
     together too — a bigger stagger spreads both launch AND finish/sync
     timing across the batch, trading total ramp-up time for a lower peak.
  Check `sweep_logs*/agent-<i>.log` for an agent that shows zero
  `Starting Run` lines — `grep -c "Starting Run" sweep_logs*/agent-*.log` —
  to tell which of the two hit.
- **A trial's `wandb sync` segfaulting** (`Segmentation fault (core dumped)
  wandb sync ...`) even though `training job rc=0` — the exact same
  `RLIMIT_NPROC` pressure, but at the *end* of a trial instead of agent
  launch: `wandb sync`'s import chain pulls in numpy, whose OpenBLAS backend
  sizes its threadpool to the login node's full core count (128) on every
  invocation, and in parallel (`SWEEP_GPUS`) mode a whole batch of agents
  tends to finish around the same wall-clock time, so many of them call sync
  within moments of each other. `sweep_dispatch.sh` forces single-threaded
  BLAS for its own sync call (`OPENBLAS_NUM_THREADS=1` etc., same fix as
  `sync_orphaned_runs.sh` already had) to avoid this; a trial whose sync
  still fails logs a `WARNING` and is picked up by `sync_orphaned_runs.sh`
  on your next run of it.
- **Sustained agent concurrency capped well below what your GPU/QoS budget
  would allow** (confirmed 2026-08-30: ~13-20 agents on one login node,
  regardless of the launch-stagger fix above) — `RLIMIT_NPROC` is enforced
  **per login node**, and Leonardo has (at least) four independent ones
  (`login01/02/05/07-ext.leonardo.cineca.it`) behind the round-robin
  `login.leonardo.cineca.it` alias a plain `ssh` lands you on one of.
  Running everything from one persistent session concentrates all your
  agents' process load on that single machine's quota. There's no
  CINECA-native mechanism for a long-lived online sweep agent at all — their
  own AI-workloads guidance only covers `WANDB_MODE=offline` +
  `wandb sync`, explicitly calling Sweeps unsupported on Leonardo given
  compute nodes' no-internet policy — so `hpc/leonardo/start_on_node.sh`
  (splitting agent batches across login nodes, same sweep_id, independent
  per-node PID/log bookkeeping via `SWEEP_NAME`) is the practical lever,
  not a deviation from some better-supported path. See its header comment
  for usage; each node still needs its own `tmux`/`screen`.
- **The same run_id gets dispatched more than once** (confirmed 2026-08-30 on
  the `parallel-2` sweep: `grep -h "trial run_id=" sweep_logs-parallel-2/agent-*.log`
  showed several run_ids repeated 2-3x *across different agents*, real
  GPU-hours wastefully redone for the identical hyperparameters) — a known,
  unresolved upstream wandb limitation for offline+SLURM sweeps (see
  [community.wandb.ai/t/.../5791](https://community.wandb.ai/t/the-sweep-agent-keeps-the-same-hyperparameters-and-run-id-in-offline-mode/5791)),
  not a bug in this repo's scripts. Mechanism: `wandb.init()` for the real
  training run doesn't happen until deep inside the offline `sbatch` job,
  hours later, and even then never touches the network (`WANDB_MODE=offline`)
  — so the sweep *controller* never learns a run_id/config was claimed until
  the eventual `wandb sync`. Any other idle agent asking for "next run" in
  that window can be handed the same still-apparently-unclaimed run_id.
  Two mitigations, can use either or both:
  - **Reduce `N`** to roughly your actually-sustainable concurrent-trial
    count (QoS/GPU budget, not wishful thinking) — fewer idle agents polling
    means fewer chances to collide with an in-flight-but-server-invisible
    run. Prefer a higher `COUNT` over a higher `N` to hit your total trial
    target.
  - **The online "claim" ping** (`sweep_dispatch.sh`, added 2026-08-30,
    marked EXPERIMENTAL in its own comment): right after receiving a
    run_id, while still on the login node (real internet), it creates that
    run online (`wandb.init(id=..., resume="allow", mode="online")`) and
    immediately `wandb.finish()`s it with zero data logged, before ever
    calling `sbatch`. This should make the run_id look "claimed" (a real,
    if empty, run object now exists) rather than merely "proposed", so the
    controller shouldn't hand it out again. The real training run resumes
    it later (`eeg_bold_fit_sweep.py`'s `wandb.init(id=run_id,
    resume="allow", ...)` already does this whenever `WANDB_RUN_ID` is
    set) and appends its actual data — resuming a run that already exists,
    even one already marked "finished", is the normal supported wandb
    resume path. Non-fatal by design: a failed claim ping just logs a
    `WARNING` and dispatch proceeds anyway. **Verify it's actually helping**
    the same way the original problem was found — after a batch of trials,
    `grep -h "trial run_id=" sweep_logs*/agent-*.log | sort | uniq -c | sort -rn`
    should show far fewer (ideally zero) counts >1.

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
