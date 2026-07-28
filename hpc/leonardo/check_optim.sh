#!/bin/bash
###############################################################################
# Preflight check for the LEONARDO EEG+BOLD optimization run. Run on a LOGIN
# node BEFORE `submit_optim.sh smoke|pilot|run`. Same intent as
# check_leonardo.sh (the reconstruction preflight): catch the "queued for
# hours, then died in 10s on a typo / missing file" failure mode by verifying
# everything the job assumes:
#   - pixi is installed and the env is built (setup_optim_env.sh has run)
#   - the target subject actually has the derivatives this stage reads
#     (EEG chunks, fMRI/BOLD, the requested leadfield) -- via the real
#     parrot_neuro.Subject code, not a hand-duplicated path guess
#   - the output area is writable
#   - the account is known to saldo
#
# Configure via env vars, or fill hpc/leonardo/config.local.sh once (same file
# the reconstruction scripts use -- WORKDIR/BIDS/REPO are shared):
#   ACCT=<...> SUBJECT=010002 bash hpc/leonardo/check_optim.sh
#
# Exits non-zero if any [FAIL] is printed.
###############################################################################
set -uo pipefail   # NOT -e: we want every check to run and report, not abort early

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for _c in "${PARROT_CONFIG:-}" "$HERE/config.local.sh" \
          "${SLURM_SUBMIT_DIR:-}/hpc/leonardo/config.local.sh" \
          "$HOME/parrot-neuro/hpc/leonardo/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; echo "[config] loaded $_c"; break; }
done

ACCT="${ACCT:-<YOUR_ACCOUNT>}"
WORKDIR="${WORKDIR:-${WORK:-}}"
: "${WORKDIR:?set WORKDIR (or export \$WORK) -- e.g. /leonardo_work/<ACCT>}"
REPO="${REPO:-$HOME/parrot-neuro}"
BIDS="${BIDS:-$WORKDIR/parrot/bids}"
SUBJECT="${SUBJECT:-010002}"
OPTIM_OUTPUT_DIR="${OPTIM_OUTPUT_DIR:-$WORKDIR/parrot/eeg_bold_fit_res}"

# Must match config.BoldFitConfig defaults / examples/eeg_bold_fit_cli.py's own
# argparse defaults -- override here (or via env) if your run uses different ones.
OPTIM_ATLAS="${OPTIM_ATLAS:-1000}"
OPTIM_SPACING="${OPTIM_SPACING:-2.0}"
OPTIM_LEADFIELD_LABEL="${OPTIM_LEADFIELD_LABEL:-duneuroCGAL}"
OPTIM_EEG_TASK="${OPTIM_EEG_TASK:-eyesclosed}"
OPTIM_FMRI_TASK="${OPTIM_FMRI_TASK:-rest}"
OPTIM_OPTIMIZE="${OPTIM_OPTIMIZE:-both}"

echo "== resolved paths (must match the job's) =="
printf '  %-20s %s\n' WORKDIR "$WORKDIR" BIDS "$BIDS" REPO "$REPO" \
       OPTIM_OUTPUT_DIR "$OPTIM_OUTPUT_DIR" SUBJECT "$SUBJECT" \
       OPTIM_ATLAS "$OPTIM_ATLAS" OPTIM_LEADFIELD "$OPTIM_LEADFIELD_LABEL-${OPTIM_SPACING}mm"

fail=0
ok()   { printf '  [ OK ]  %s\n' "$1"; }
warn() { printf '  [WARN]  %s\n' "$1"; }
bad()  { printf '  [FAIL]  %s\n' "$1"; fail=1; }

echo "== pixi environment =="
PIXI="$(command -v pixi || true)"
[ -z "$PIXI" ] && [ -x "$HOME/.pixi/bin/pixi" ] && PIXI="$HOME/.pixi/bin/pixi"
if [ -z "$PIXI" ]; then
  bad "pixi not on PATH and not at \$HOME/.pixi/bin/pixi -- run setup_optim_env.sh"
else
  ok "pixi $("$PIXI" --version 2>/dev/null) ($PIXI)"
fi
if [ ! -d "$REPO/.pixi" ]; then
  bad ".pixi env dir missing at $REPO/.pixi -- run: bash hpc/leonardo/setup_optim_env.sh"
elif [ -n "$PIXI" ]; then
  if (cd "$REPO" && "$PIXI" run python -c "import jax, tvboptim, optax, equinox" >/dev/null 2>&1); then
    ok "env imports jax/tvboptim/optax/equinox"
  else
    bad "env exists but import check failed -- re-run setup_optim_env.sh (see its output for the error)"
  fi
fi

echo "== repo / driver script =="
[ -f "$REPO/examples/eeg_bold_fit_cli.py" ] && ok "examples/eeg_bold_fit_cli.py present" \
  || bad "examples/eeg_bold_fit_cli.py missing in $REPO"

echo "== subject derivatives: sub-$SUBJECT (via parrot_neuro.Subject) =="
if [ -n "$PIXI" ] && [ -d "$REPO/.pixi" ]; then
  (cd "$REPO" && "$PIXI" run python - "$BIDS" "$SUBJECT" "$OPTIM_ATLAS" "$OPTIM_SPACING" \
      "$OPTIM_LEADFIELD_LABEL" "$OPTIM_EEG_TASK" "$OPTIM_FMRI_TASK" "$OPTIM_OPTIMIZE" <<'PY'
import sys
bids, subj, atlas, spacing, lf_label, eeg_task, fmri_task, optimize = sys.argv[1:9]
atlas = int(atlas)

def report(ok, msg):
    print(f"  [{'OK' if ok else 'FAIL'}]  {msg}")
    return ok

fail = False
try:
    from parrot_neuro import Subject
    s = Subject(bids, subj)
except Exception as e:
    report(False, f"Subject({bids!r}, {subj!r}) failed to construct: {e}")
    sys.exit(1)
fail |= not report((s.bids_root / "derivatives").is_dir(), "derivatives/ tree present")

lf_key = f"{lf_label}-{spacing}mm"
lf_path = s.path.leadfield(lf_key)
fail |= not report(lf_path.exists(), f"leadfield [{lf_key}] present ({lf_path})")

if optimize != "bold":
    eeg_path = s.path.eeg(eeg_task)
    eeg_ok = eeg_path.exists()
    fail |= not report(eeg_ok, f"EEG derivatives [task-{eeg_task}] present ({eeg_path})")
    if eeg_ok:
        sidecar = eeg_path.with_suffix(".json")
        fail |= not report(sidecar.exists(), f"EEG sidecar JSON present ({sidecar})")
else:
    print(f"  [ -- ]  optimize={optimize!r}: EEG not required, skipping EEG checks")

try:
    nodes = s.load.fmri_nodes(atlas, fmri_task)
    report(True, f"fMRI/BOLD [atlas-{atlas}, task-{fmri_task}] loads ({nodes.keep.sum()} nodes kept)")
except Exception as e:
    fail = True
    report(False, f"fMRI/BOLD [atlas-{atlas}, task-{fmri_task}] failed to load: {e}")

sys.exit(1 if fail else 0)
PY
  ) || fail=1
else
  bad "skipped (pixi env not ready -- see above)"
fi

echo "== output area =="
if mkdir -p "$OPTIM_OUTPUT_DIR" 2>/dev/null && t="$(mktemp "$OPTIM_OUTPUT_DIR/.parrot_optim_check.XXXXXX" 2>/dev/null)"; then
  rm -f "$t"; ok "writable: $OPTIM_OUTPUT_DIR"
else
  bad "cannot create/write to $OPTIM_OUTPUT_DIR (wrong path, or quota?)"
fi

echo "== account / budget =="
if command -v saldo >/dev/null 2>&1; then
  if saldo -b 2>/dev/null | grep -q "$ACCT"; then ok "$ACCT found in 'saldo -b'"; else warn "$ACCT not in 'saldo -b' -- check the account name"; fi
else
  warn "saldo not on PATH -- skipping budget check"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "PREFLIGHT PASSED -- safe to: bash hpc/leonardo/submit_optim.sh smoke"
else
  echo "PREFLIGHT FAILED -- fix the [FAIL] items above before submitting."
  exit 1
fi
