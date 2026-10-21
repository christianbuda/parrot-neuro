# Quality-control reports

Rebuild after editing the QC package: `./bin/build.sh qc`.
To refresh existing derivatives without rerunning reconstruction:

```bash
./bin/run_reconstruction.sh <bids_dir> <output_dir> participant --stages qc
```

The subject report is `qc/sub-<ID>/index.html`; the cohort index is `qc/index.html`.

## Status meanings

- **PASS**: automated checks passed; visual review is still needed.
- **WARN**: review the named output or limitation.
- **FAIL**: an output failed a quality check, not necessarily a pipeline crash.
- **SKIP**: not produced/not applicable. Missing outputs alone cannot distinguish an
  unrequested stage from an interrupted one.
- **CRITICAL FAILURE**: a pipeline-stopping error was recorded. The subject banner
  and group row identify the stage, exit code, timestamp, and log.

OpenMEEG/BEM is optional. Original-shell intersections remain warnings and solver
availability reports the actual outcome; missing BEM does not invalidate FEM.
Limited neck coverage is informational when artifact sources use the template block;
the calibration and source gains are checked separately. Electrode clearance is a
warning indicator; pathological leadfield gains can still fail QC.

## Stopped runs

The orchestrator writes `logs/sub-<ID>/pipeline_failure.txt` before attempting a
lightweight report (no imaging checks or rendering). The marker remains readable by
later QC runs even if the QC container could not launch. A successful retry selecting
the failed reconstruction stage clears it; running only QC does not clear a
reconstruction failure. Existing failed logs are retained as history.

SIGKILL, node loss, and older runs cannot reliably be identified from these markers.
Check scheduler status and report timestamps in those cases. Full QC of partial
outputs can be run separately; it preserves the critical banner while the marker exists.

Run reporting regression tests inside the QC image with the repository mounted at
`/repo`: `python /repo/tests/test_qc_reporting.py`.
