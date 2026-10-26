"""QC regression tests; run inside the QC image (requires its imaging dependencies)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "containers" / "parrot_qc"))
sys.path.insert(0, str(ROOT / "containers" / "_shared"))

from qc.checks import StageResult
from qc.context import Context
from qc.report import write_failure_report, write_group_report, write_subject_report
from qc.stages import artifacts, leadfields


class QCReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ctx = Context(self.root, "01")

    def marker(self):
        path = self.root / "logs/sub-01/pipeline_failure.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("forwardsolvers\n42\n2026-10-13T00:00:00\nlogs/failed.txt\n")
        return path

    def test_failure_overrides_stale_pass_and_survives_qc(self):
        result = StageResult("test", "Test")
        result.ok("valid")
        write_subject_report(self.ctx, [result])
        marker = self.marker()
        self.assertEqual(write_group_report(self.root), "critical")
        self.assertEqual(write_subject_report(self.ctx, [result]), "critical")
        self.assertIn("CRITICAL FAILURE", (self.ctx.out_dir / "index.html").read_text())
        marker.unlink()  # successful reconstruction retry
        self.assertEqual(write_subject_report(self.ctx, [result]), "pass")
        self.assertEqual(write_group_report(self.root), "pass")

    def test_failure_without_prior_report(self):
        self.marker()
        self.assertEqual(write_group_report(self.root), "critical")
        html = (self.root / "qc/index.html").read_text()
        self.assertIn("sub-01", html)
        self.assertNotIn('href="sub-01/index.html"', html)
        write_failure_report(self.root, "sub-01")
        report = json.loads((self.ctx.out_dir / "qc_report.json").read_text())
        self.assertEqual(report["execution_failure"]["exit_code"], 42)
        self.assertEqual(report["stages"], [])

    def test_optional_bem_missing_and_artifacts_not_brain_leadfields(self):
        folder = self.ctx.stage_dir("leadfields")
        folder.mkdir(parents=True)
        np.save(folder / "processed_duneuro_artifact-eyes-CGAL-leadfield.npy", np.ones((2, 3)))
        result = leadfields.run(self.ctx)
        self.assertEqual(result.status, "skip")
        np.save(folder / "processed_duneuroCGAL-2.0mm-leadfield.npy", np.ones((2, 3)))
        with patch.object(self.ctx, "add_figure"):
            result = leadfields.run(self.ctx)
        checks = {c.name: c.status for c in result.checks}
        self.assertEqual(checks["OpenMEEG availability"], "skip")
        self.assertEqual(checks["DUNEuro CGAL availability"], "pass")
        self.assertEqual(checks["DUNEuro CGAL_anisotropic availability"], "skip")
        self.assertEqual(result.status, "pass")

    def test_clearance_warns_but_numerical_failure_still_fails(self):
        result = StageResult("artifacts", "Artifacts")
        artifacts._check_electrode_clearance(
            result, "muscle", np.array([[2.94, 0, 0]]), np.zeros((1, 3)), 5)
        self.assertEqual(result.status, "warn")
        matrix = np.ones((2, 3000))
        matrix[:, :3] = 100
        artifacts._check_source_outliers(result, "muscle", matrix)
        self.assertEqual(result.status, "fail")

    def test_exit_trap_records_failure_even_if_reporting_fails(self):
        script = (ROOT / "bin/run_reconstruction.sh").read_text()
        cleanup = script.split("cleanup() {", 1)[1].split("\n}", 1)[0]
        shell = '\n'.join([
            'OUTPUT_DIR="$1"', 'CURRENT_SUBJECT=01', 'NAME=tetmesh',
            'FAILURE_EXIT_CODE=42', 'FAILURE_LOG=failed.log',
            'run_in_docker_QC() { return 99; }',
            'cleanup() {' + cleanup + '\n}', 'trap cleanup EXIT', 'exit 1',
        ])
        proc = subprocess.run(["bash", "-c", shell, "test", str(self.root)])
        self.assertEqual(proc.returncode, 1)
        lines = (self.root / "logs/sub-01/pipeline_failure.txt").read_text().splitlines()
        self.assertEqual(lines[:2], ["tetmesh", "42"])
        self.assertEqual(lines[3], "failed.log")

    def test_qc_only_keeps_failure_but_stage_retry_clears_it(self):
        script = (ROOT / "bin/run_reconstruction.sh").read_text()
        block = script.split('    if [ -f "$LOG_DIR/pipeline_failure.txt" ]; then', 1)[1]
        block = 'if [ -f "$LOG_DIR/pipeline_failure.txt" ]; then' + block.split('\n    NAME="qc"', 1)[0]
        for selected, remains in (("qc", True), ("tetmesh", True), ("forwardsolvers", False)):
            with self.subTest(selected=selected):
                marker = self.marker()
                shell = '\n'.join([
                    'LOG_DIR="$1"', 'STAGE_SET=("$2")',
                    'want_stage() { printf "%s\\n" "${STAGE_SET[@]}" | grep -qx -- "$1"; }', block,
                ])
                subprocess.run(["bash", "-c", shell, "test", str(marker.parent), selected], check=True)
                self.assertEqual(marker.exists(), remains)

    def test_failure_cli_does_not_import_rendering(self):
        self.marker()
        script = '''
import sys
from qc.run_qc import main
sys.argv = ['qc', '--failure-only', '--subject', '01', '--output_dir', sys.argv[1]]
main()
assert 'pyvista' not in sys.modules
assert 'qc.stages' not in sys.modules
'''
        import os
        env = dict(os.environ, PYTHONPATH=str(ROOT / "containers/parrot_qc"))
        subprocess.run([sys.executable, "-c", script, str(self.root)], env=env, check=True)


if __name__ == "__main__":
    unittest.main()
