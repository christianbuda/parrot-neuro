# Running Parrot's EEG+BOLD optimization stage on JUPITER (JSC)

This directory is the JUPITER analogue of `hpc/leonardo/`'s **EEG+BOLD
optimization stage** section only (`parrot_neuro.optimization`, driven by
`examples/eeg_bold_fit_cli.py`) -- **not** the reconstruction pipeline (the
four Docker/Apptainer images). There is no JUPITER port of reconstruction
here: this assumes subjects are already reconstructed elsewhere (e.g.
LEONARDO) and their derivatives (leadfield, EEG, fMRI/BOLD) are staged onto
JUPITER's storage before running anything in this directory. See
`hpc/leonardo/README.md` for the fit hyperparameters themselves, the
Optuna/wandb hyperparameter-search workflow, and the full rationale behind
each `OPTIM_*` knob -- all of that carries over unchanged; only the cluster
plumbing (paths, SLURM resources, environment setup) differs, documented here.

## Why a separate directory, not a flag on `hpc/leonardo/`

JUPITER (`login.jupiter.fz-juelich.de`) is JSC's EuroHPC exascale system --
architecturally different enough from CINECA LEONARDO that reusing the same
scripts with a few `if`s would be more confusing than a parallel, mirrored
set of files:

| | LEONARDO (Booster) | JUPITER |
|---|---|---|
| GPU | 4x NVIDIA A100 / node | 4x NVIDIA GH200 Grace-Hopper superchip / node |
| CPU arch | x86_64 (`linux-64`) | **aarch64** (`linux-aarch64`) -- Grace is ARM |
| `--gres` syntax | untyped: `--gres=gpu:1` | **typed**: `--gres=gpu:gh200:1` |
| Confirmed partition | `boost_usr_prod` | `booster` (also `largebooster`) |
| Confirmed QoS | `normal` (24h), `boost_qos_dbg` (30m), `boost_qos_lprod` (4d) | `normal` (**12h** cap) -- no dedicated debug/long QoS confirmed on this account (see below) |
| Storage root | `/leonardo_work/<ACCT>` | `/p/project1/<...>`, `/p/scratch/<...>`, `/p/data1/<...>` |
| Budget tool | `saldo -b` | `jutil user projects` |
| Container runtime | Singularity (system command) | Apptainer + Singularity (both system commands, confirmed at `/usr/bin/`) |

Confirmed 2026-09 on this account (`gambosi1`, project `e-aif-2026fl01-991`,
PI `toschi3`) via `sinfo` / `scontrol show node` / `sacctmgr show qos`:
`ClusterName=jupiter`, partition `booster`, `Gres=gpu:gh200:4` per node,
288 cores / ~878G RAM per node (so ~72 cores / ~219G per GPU-share), QoS
`normal` has `MaxWall=12:00:00`. `lowprio`/`highprio` also cap at 12h;
`nolimits`/`part_boos+`/`part_larg+`/`jschealth` exist cluster-wide but were
**not** confirmed usable by this account -- before assuming a longer QoS is
available, check:
```bash
sacctmgr show assoc user=$USER format=Account,Partition,QOS%60
```

## *** Known blocker: pixi.toml doesn't support linux-aarch64 yet ***

JUPITER's GH200 nodes are `linux-aarch64`. The repo's `pixi.toml` currently
declares `platforms = ["linux-64", "osx-arm64"]` only -- `linux-aarch64` was
tried before and dropped because conda-forge has no `marimo` build for it
(see `pixi.toml`'s own comment). `marimo`/`ipykernel` are dev-only (the local
notebooks under `development/`) -- `eeg_bold_fit_cli.py` never imports them.

**The fix, not yet done**: split `pixi.toml` into a `dev` feature
(`marimo`/`ipykernel`, `linux-64`/`osx-arm64` only) and an `optim` feature
(`jax`/`optax`/`equinox`/`tvboptim`/`mne`/etc. -- everything the fit actually
uses) that also targets `linux-aarch64`, via pixi's `[feature.*]`/
`[environments]` tables. Then `pixi install -e optim`, run on JUPITER's login
node (real internet access, same assumption as LEONARDO's login nodes),
resolves/locks the aarch64 env against conda-forge/PyPI. This has **not**
been done yet -- `setup_optim_env.sh` here already assumes the `optim`
environment name and will fail loudly (with a pointer back to this note) if
`pixi.toml` hasn't been restructured. Do that first, then proceed below.

One more unknown worth flagging once that's unblocked: `jax[cuda12]`'s pip
wheels need to actually ship `manylinux_aarch64` builds for the pinned JAX
version -- verify this as part of the `pixi install -e optim` step (it will
simply fail to resolve if not), don't assume it from the LEONARDO/x86_64
experience.

## Setup, in order

```bash
# 1. Clone the repo (if not already) and set your account/paths ONCE:
cp hpc/jupiter/config.local.sh.example hpc/jupiter/config.local.sh
# edit ACCT (this account: e-aif-2026fl01-991) and WORKDIR (find your actual
# project directory: ls /p/project1/ | grep -i aif , or /p/scratch/, /p/data1/)

# 2. Stage this subject's already-reconstructed derivatives onto JUPITER's
#    storage (rsync/rclone from wherever reconstruction ran, e.g. LEONARDO) --
#    not scripted here; layout must match $BIDS/derivatives same as LEONARDO.

# 3. Resolve the pixi.toml linux-aarch64 blocker above (one-time, touches the
#    shared pixi.toml -- do this deliberately, not as a side effect of
#    following this README on autopilot).

# 4. Build the pixi env (needs internet -- login node):
bash hpc/jupiter/setup_optim_env.sh

# 5. Preflight:
bash hpc/jupiter/check_optim.sh

# 6. Smoke test, then pilot (measure before scaling -- same discipline as
#    LEONARDO's README), then the real run:
bash hpc/jupiter/submit_optim.sh smoke        # ~minutes
bash hpc/jupiter/submit_optim.sh pilot        # full hyperparameters, ONE subject, timed
squeue --me
```

Read the pilot's `.out` for `optim finished in N min, rc=0` and the GPU-idle
summary. Set `OPTIM_TIME`/`OPTIM_MEM`/`OPTIM_CPUS` in `config.local.sh` from
what you observe (this folder's defaults -- `10:00:00`/`128G`/`16` cores --
are UNMEASURED placeholders, sized conservatively against one GH200
GPU-share's ceiling, not validated numbers). **`OPTIM_TIME` must stay under
the confirmed 12h `normal` QoS cap** -- if a subject's fit doesn't fit,
either enable `OPTIM_EARLY_STOP_PATIENCE`, reduce `OPTIM_NUM_EPOCHS`, or
confirm a longer QoS is actually usable on this account first (see the QoS
table above).

## Running a subgroup once you have hyperparameters picked out

`subj_id.txt` in this directory (copied from `hpc/leonardo/subj_id.txt`) is
a whitespace-separated subject-label list -- shell word-splitting on the tabs
means it feeds directly into `submit_optim.sh run`'s positional-args form:

```bash
bash hpc/jupiter/submit_optim.sh list                 # sanity-check the values got picked up
bash hpc/jupiter/submit_optim.sh run $(cat hpc/jupiter/subj_id.txt)
```

Each subject still gets its own full-GPU array task, submitted as one
`sbatch` call, `ARRAY_THROTTLE`-capped concurrency (default `%40`) -- same
mechanism as `hpc/leonardo/submit_optim.sh run`, see its README section for
the full rationale. Set the fit hyperparameters (the 7 Optuna/wandb-swept
fields plus whatever else your search fixed -- `OPTIM_ATLAS`,
`OPTIM_SCHEDULE`, `OPTIM_BOLD_MODEL`, etc.) in `config.local.sh` first.

## Notes / gotchas specific to JUPITER

- **No JUPITER reconstruction pipeline here.** If a subject hasn't been
  reconstructed yet, run that stage elsewhere (e.g. LEONARDO via
  `hpc/leonardo/`) and stage the resulting `derivatives/` tree onto JUPITER's
  storage before using anything in this directory.
- **Compute-node internet access is unverified for JUPITER** -- this
  directory assumes the same LEONARDO-style split (login nodes have egress,
  compute nodes don't) as the safe default. If JUPITER's compute nodes
  actually do have internet, nothing here breaks, it's just unnecessarily
  cautious; if you confirm otherwise, no change needed.
- **GH200 GPU-memory numbers are unverified.** The OOM-avoidance defaults
  (`OPTIM_T1_WARMUP=30000`, `OPTIM_SOLVER_BLOCK_SIZE=1400`) are carried over
  from LEONARDO's A100 tuning (see `hpc/leonardo/README.md`'s OOM notes) --
  GH200's unified CPU/GPU memory (NVLink-C2C) is a different memory model
  entirely; re-measure with `pilot` rather than assuming these numbers (or
  the ~31GiB peak LEONARDO measured) carry over.
- **`jutil user projects`** lists your project/budget name(s) -- use this
  instead of LEONARDO's `saldo -b` wherever these scripts check the account.
