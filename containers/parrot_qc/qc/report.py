"""Assemble per-subject and group QC outputs: index.html (Jinja2) + qc_report.json.

The JSON is the machine-readable record (one object per stage with its checks and
figure paths); index.html is the human view embedding the figures and a colour
status table. The group index aggregates every subject's JSON into a stage matrix.
"""
from __future__ import annotations

import datetime
import json
import os
import stat
import uuid
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .checks import worst, SKIP

_TEMPLATES = Path(__file__).parent / "templates"


def _write(path, text):
    """Publish complete files so concurrent group scans cannot read partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # Exclusive creation uses normal permissions filtered by the process umask.
        with temporary.open("x") as f:
            f.write(text)
        if path.exists():
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def execution_failure(deriv, subject):
    marker = Path(deriv) / "logs" / subject / "pipeline_failure.txt"
    if not marker.exists():
        return None
    stage, code, timestamp, log = marker.read_text().splitlines()[:4]
    return {"stage": stage, "exit_code": int(code), "timestamp": timestamp, "log": log}


def write_failure_report(deriv, subject):
    """Render a stopped run without importing stage checkers or loading imaging data."""
    failure = execution_failure(deriv, subject)
    if failure is None:
        raise ValueError(f"No pipeline failure recorded for {subject}")
    payload = {"subject": subject, "generated": datetime.datetime.now().isoformat(timespec="seconds"),
               "overall_status": "critical", "execution_failure": failure, "stages": []}
    out = Path(deriv) / "qc" / subject
    _write(out / "qc_report.json", json.dumps(payload, indent=2))
    _write(out / "index.html", _env().get_template("subject.html.j2").render(report=payload))
    write_group_report(deriv)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=True,
    )


def _result_to_dict(r) -> dict:
    return {
        "name": r.name,
        "title": r.title,
        "present": r.present,
        "status": r.status,
        "description": r.description,
        "notes": r.notes,
        "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in r.checks],
        "figures": [{"caption": cap, "path": path} for cap, path in r.figures],
    }


def write_subject_report(ctx, results) -> str:
    """Write qc_report.json + index.html for one subject; return overall status."""
    overall = worst(r.status for r in results)
    failure = execution_failure(ctx.deriv, ctx.subj)
    if failure:
        overall = "critical"
    payload = {
        "subject": ctx.subj,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "overall_status": overall,
        "execution_failure": failure,
        "stages": [_result_to_dict(r) for r in results],
    }
    _write(ctx.out_dir / "qc_report.json", json.dumps(payload, indent=2))

    html = _env().get_template("subject.html.j2").render(report=payload)
    _write(ctx.out_dir / "index.html", html)
    return overall


def write_group_report(deriv) -> str:
    """Scan qc/sub-*/qc_report.json and write the group index.html. Returns status."""
    qc_root = Path(deriv) / "qc"
    subjects = []
    stage_titles: dict[str, str] = {}
    subject_ids = {p.parent.name for p in qc_root.glob("sub-*/qc_report.json")}
    subject_ids.update(p.parent.name for p in (Path(deriv) / "logs").glob("sub-*/pipeline_failure.txt"))
    for subject in sorted(subject_ids):
        jf = qc_root / subject / "qc_report.json"
        data = json.loads(jf.read_text()) if jf.exists() else {"subject": subject, "stages": [], "overall_status": SKIP}
        failure = execution_failure(deriv, subject)
        per_stage = {}
        for s in data["stages"]:
            per_stage[s["name"]] = s["status"]
            stage_titles.setdefault(s["name"], s["title"])
        subjects.append({
            "subject": data["subject"],
            "overall": "critical" if failure else data["overall_status"],
            "execution_failure": failure,
            "stages": per_stage,
            "href": f"{data['subject']}/index.html" if jf.exists() else None,
        })

    stage_order = list(stage_titles.keys())
    overall = worst([s["overall"] for s in subjects]) if subjects else SKIP
    html = _env().get_template("group.html.j2").render(
        subjects=subjects,
        stage_order=stage_order,
        stage_titles=stage_titles,
        generated=datetime.datetime.now().isoformat(timespec="seconds"),
        overall=overall,
    )
    _write(qc_root / "index.html", html)
    return overall
