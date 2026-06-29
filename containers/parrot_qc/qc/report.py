"""Assemble per-subject and group QC outputs: index.html (Jinja2) + qc_report.json.

The JSON is the machine-readable record (one object per stage with its checks and
figure paths); index.html is the human view embedding the figures and a colour
status table. The group index aggregates every subject's JSON into a stage matrix.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .checks import worst, SKIP

_TEMPLATES = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html"]),
    )


def _result_to_dict(r) -> dict:
    return {
        "name": r.name,
        "title": r.title,
        "present": r.present,
        "status": r.status,
        "notes": r.notes,
        "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in r.checks],
        "figures": [{"caption": cap, "path": path} for cap, path in r.figures],
    }


def write_subject_report(ctx, results) -> str:
    """Write qc_report.json + index.html for one subject; return overall status."""
    overall = worst(r.status for r in results)
    payload = {
        "subject": ctx.subj,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "overall_status": overall,
        "stages": [_result_to_dict(r) for r in results],
    }
    (ctx.out_dir / "qc_report.json").write_text(json.dumps(payload, indent=2))

    html = _env().get_template("subject.html.j2").render(report=payload)
    (ctx.out_dir / "index.html").write_text(html)
    return overall


def write_group_report(deriv) -> str:
    """Scan qc/sub-*/qc_report.json and write the group index.html. Returns status."""
    qc_root = Path(deriv) / "qc"
    subjects = []
    stage_titles: dict[str, str] = {}
    for jf in sorted(qc_root.glob("sub-*/qc_report.json")):
        data = json.loads(jf.read_text())
        per_stage = {}
        for s in data["stages"]:
            per_stage[s["name"]] = s["status"]
            stage_titles.setdefault(s["name"], s["title"])
        subjects.append({
            "subject": data["subject"],
            "overall": data["overall_status"],
            "stages": per_stage,
            "href": f"{data['subject']}/index.html",
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
    (qc_root / "index.html").write_text(html)
    return overall
