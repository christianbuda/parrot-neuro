"""Entry point for the Parrot final-QC stage.

Per subject: run every stage checker (each robust -- a stage crash becomes a
single FAIL check, never aborting the report), then write qc/sub-<id>/index.html
+ qc_report.json. With --group: aggregate all subjects into qc/index.html.

Invoked inside the parrot_qc image as:
    python /qc/run_qc.py --subject <id> --output_dir /derivatives [--threads N]
    python /qc/run_qc.py --group        --output_dir /derivatives

Always exits 0 on a successful *run* even if checks fail -- QC informs, it does
not block the pipeline. Non-zero exit means the QC machinery itself broke.
"""
import argparse
import os
import sys
import tempfile
import traceback

# Make the bind-mounted package importable when run as a plain script.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# Writable matplotlib cache (HOME is the managed /parrot_home, but be defensive).
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
os.environ.setdefault("MPLBACKEND", "Agg")

from qc.checks import StageResult            # noqa: E402
from qc.context import Context               # noqa: E402
from qc.report import write_subject_report, write_group_report  # noqa: E402
from qc.stages import STAGES                 # noqa: E402


def run_subject(deriv: str, subject: str) -> str:
    ctx = Context(deriv, subject)
    results = []
    for mod in STAGES:
        try:
            r = mod.run(ctx)
        except Exception as e:  # noqa: BLE001 - one bad stage must not sink the report
            r = StageResult(mod.NAME, mod.TITLE)
            r.fail("stage error", f"{type(e).__name__}: {e}")
            traceback.print_exc()
        results.append(r)
        print(f"  [{r.status:>4}] {r.title}", flush=True)
    overall = write_subject_report(ctx, results)
    print(f"\nsub-{subject}: overall = {overall.upper()}")
    print(f"report -> {ctx.out_dir / 'index.html'}")
    # Keep the group index in sync after a standalone subject rerun. The
    # orchestrator runs a dedicated --group pass after the subject loop, but a
    # manual per-subject rerun would otherwise leave qc/index.html stale (still
    # showing the previous run's status). Cheap (JSON re-scan) and never fatal.
    try:
        write_group_report(deriv)
    except Exception:  # noqa: BLE001 - a group refresh failure must not sink the subject run
        traceback.print_exc()
    return overall


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", help="bare subject id, e.g. 010002")
    p.add_argument("--output_dir", required=True, help="derivatives root")
    p.add_argument("--group", action="store_true", help="write the group index instead")
    p.add_argument("--threads", type=int, default=1, help="accepted for orchestrator parity")
    args = p.parse_args()

    if args.group:
        overall = write_group_report(args.output_dir)
        print(f"group QC: overall = {overall.upper()} -> "
              f"{os.path.join(args.output_dir, 'qc', 'index.html')}")
        return

    if not args.subject:
        p.error("--subject is required unless --group is given")
    run_subject(args.output_dir, args.subject)


if __name__ == "__main__":
    main()
