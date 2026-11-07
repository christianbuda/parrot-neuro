"""Manifest inventory. Paths remain relative so datasets can be relocated."""
from __future__ import annotations

import json
from pathlib import Path


class OutputInventory:
    def __init__(self, deriv: Path, subj: str):
        self.deriv = deriv
        self.directory = deriv / "manifests" / subj
        self.entries = []
        for manifest in sorted(self.directory.glob("*.json")):
            data = json.loads(manifest.read_text())
            for entry in data["outputs"]:
                relative = Path(entry["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Invalid relative output path in {manifest}: {relative}")
                self.entries.append({**entry, "producer_tag": data["producer_tag"]})

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"OutputInventory({len(self)} outputs, {self.directory})"

    def find(self, kind: str | None = None, **parameters) -> list[dict]:
        return [e for e in self if (kind is None or e["type"] == kind)
                and all(e.get("parameters", {}).get(k) == v for k, v in parameters.items())]

    def path(self, kind: str, **parameters) -> Path:
        matches = self.find(kind, **parameters)
        if not matches:
            raise FileNotFoundError(
                f"Output {kind!r} {parameters} is not registered in {self.directory}. "
                "Existing datasets need manifest population separately."
            )
        paths = {e["path"] for e in matches}
        if len(paths) != 1:
            raise ValueError(f"Ambiguous output {kind!r}: specify parameters ({parameters})")
        path = self.deriv / paths.pop()
        if not path.exists():
            raise FileNotFoundError(f"Registered output is missing: {path}")
        return path
