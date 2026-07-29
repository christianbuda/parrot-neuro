"""Render figures + a self-contained HTML/JSON report for one subject's
channel QC, across however many tasks were run. Mirrors the shape of
``containers/parrot_qc/qc/report.py`` (JSON = machine record, HTML = human
view over the same data + embedded figures) without depending on that
package (separate image, not importable here).
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import viz
from .checks import FAIL, PASS, WARN, worst
from .flags import ChannelFlag, Thresholds, flag_channels, missing_channels
from .metrics import ChannelMetrics, compute_channel_metrics

_TEMPLATES = Path(__file__).parent / "templates"


@dataclass
class TaskQCResult:
    task: str
    channel_names: list[str]
    metrics: ChannelMetrics
    flags: list[ChannelFlag]
    missing: list[str] = field(default_factory=list)
    unpositioned: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        statuses = [f.status for f in self.flags]
        if self.missing or self.unpositioned:
            statuses.append(FAIL)
        return worst(statuses) if statuses else PASS

    def counts(self) -> dict[str, int]:
        c = {PASS: 0, WARN: 0, FAIL: 0}
        for f in self.flags:
            c[f.status] += 1
        return c


def run_task_qc(task_eeg, expected_channels=None, thresholds: Thresholds | None = None) -> TaskQCResult:
    """Compute metrics + flags for one already-loaded :class:`~.data.TaskEEG`."""
    metrics = compute_channel_metrics(
        task_eeg.segments, task_eeg.channel_names, task_eeg.sfreq, positions=task_eeg.positions
    )
    flags = flag_channels(metrics, task_eeg.positions, thresholds)
    missing = missing_channels(task_eeg.channel_names, expected_channels) if expected_channels else []
    unpositioned = [n for n in task_eeg.channel_names if n not in task_eeg.positions]
    return TaskQCResult(
        task=task_eeg.task, channel_names=task_eeg.channel_names, metrics=metrics,
        flags=flags, missing=missing, unpositioned=unpositioned,
    )


def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(_TEMPLATES)), autoescape=select_autoescape(["html"]))


def _render_task_figures(task_eeg, result: TaskQCResult, fig_dir: Path) -> list[dict]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = [
        ("montage_status", "Electrode status (scalp map)",
         lambda: viz.plot_montage_status(result.channel_names, task_eeg.positions, result.flags,
                                          missing=result.missing)),
        ("metric_topomaps", "Spatial distribution of key metrics",
         lambda: viz.plot_metric_topomaps(result.metrics, task_eeg.positions, result.flags)),
        ("psd", "Per-channel power spectral density",
         lambda: viz.plot_channel_psd(result.metrics, result.flags)),
        ("timeseries", "Raw traces (first segment)",
         lambda: viz.plot_channel_timeseries(task_eeg.segments[0], task_eeg.sfreq, result.channel_names,
                                              result.flags)),
        ("correlation", "Inter-channel correlation",
         lambda: viz.plot_correlation_matrix(task_eeg.segments, result.channel_names, result.flags)),
        ("summary", "Channels ranked by criteria triggered",
         lambda: viz.plot_summary_bar(result.metrics, result.flags)),
    ]
    entries = []
    for stem, caption, render_fn in figs:
        rel = f"figures/{result.task}_{stem}.png"
        fig = render_fn()
        fig.savefig(fig_dir.parent / rel, dpi=150)
        import matplotlib.pyplot as plt

        plt.close(fig)
        entries.append({"caption": caption, "path": rel})
    return entries


def _metrics_table(result: TaskQCResult) -> list[dict]:
    m = result.metrics
    by_name = {f.name: f for f in result.flags}
    rows = []
    for i, name in enumerate(result.channel_names):
        f = by_name[name]
        rows.append({
            "name": name, "status": f.status, "reasons": "; ".join(f.reasons),
            "rms": float(m.rms[i]), "flatline_pct": float(m.flatline_fraction[i] * 100),
            "kurtosis": float(m.kurtosis[i]), "hf_noise_ratio": float(m.hf_noise_ratio[i]),
            "line_noise_ratio": float(m.line_noise_ratio[i]), "neighbor_corr": float(m.neighbor_corr[i]),
            "segment_std_cv": float(m.segment_std_cv[i]),
        })
    rows.sort(key=lambda r: ({PASS: 0, WARN: 1, FAIL: 2}[r["status"]]), reverse=True)
    return rows


def write_subject_report(subject_id: str, results: dict[str, TaskQCResult], task_eegs: dict, out_dir: Path) -> str:
    """Render figures + write ``index.html``/``channel_qc.json`` for every
    task; returns the overall (worst-of-all-tasks) status."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks_payload = []
    for task, result in results.items():
        figures = _render_task_figures(task_eegs[task], result, out_dir / "figures")
        tasks_payload.append({
            "task": task, "status": result.status, "counts": result.counts(),
            "missing": result.missing, "unpositioned": result.unpositioned,
            "rows": _metrics_table(result), "figures": figures,
        })

    # cross-task view: same channel, different status in different tasks
    all_channels = sorted({n for r in results.values() for n in r.channel_names})
    cross_task = []
    for name in all_channels:
        row = {"name": name}
        for task, result in results.items():
            f = next((f for f in result.flags if f.name == name), None)
            row[task] = f.status if f else "missing"
        cross_task.append(row)

    overall = worst(t["status"] for t in tasks_payload) if tasks_payload else PASS
    payload = {
        "subject": subject_id,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "overall_status": overall,
        "tasks": tasks_payload,
        "task_names": list(results.keys()),
        "cross_task": cross_task,
    }
    (out_dir / "channel_qc.json").write_text(json.dumps(payload, indent=2, default=_json_default))

    html = _env().get_template("report.html.j2").render(report=payload)
    (out_dir / "index.html").write_text(html)
    return overall


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")
