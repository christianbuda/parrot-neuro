"""``NMMParams`` -- a portable neural-mass parameter set, on the connectome node
axis (M), with a subject-fit -> template -> defaults fallback chain.

numpy + stdlib only. Never import jax, tvboptim, or ``parrot_neuro.optimization``
-- see STEP1-SPEC.md's hard invariants. ``Subject`` is imported lazily (inside
``from_legacy``'s fallback branch only) so a bare ``import
parrot_neuro.simulation.params`` stays numpy+stdlib-cheap even though ``Subject``
itself pulls in nibabel/trimesh.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

import numpy as np

from .. import __version__ as _PARROT_NEURO_VERSION
from ._modelspec import ModelSpec, get_spec

if TYPE_CHECKING:  # avoid importing Subject (nibabel/trimesh) at module load time
    from ..subject import Subject

# --- provenance: a closed allowlist, enforced in NMMParams.save ------------
_PROVENANCE_ALLOWED_KEYS = frozenset({
    "kind", "chain", "coverage", "subject", "dataset", "group", "optimize",
    "epochs_run", "stopped_early", "final_loss_eeg", "final_loss_bold",
    "eeg_loss_ratio", "bold_loss_ratio", "tvboptim", "parrot_neuro",
    "git_commit", "date",
})

# The legacy npz mixes parameter arrays with bookkeeping -- excluded by this
# explicit name set, never by a shape heuristic (see from_legacy).
_LEGACY_BOOKKEEPING_KEYS = frozenset({
    "loss_eeg", "loss_bold", "eeg_loss_ratio", "bold_loss_ratio", "combined_loss_ratio",
})


@dataclass(frozen=True)
class _NameLoc:
    """Minimal duck-typed stand-in for a `LearnableParam` (name + location only)."""

    name: str
    location: str


# --- small stdlib-only helpers ----------------------------------------------
def _tvboptim_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("tvboptim")
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    """Best-effort current commit hash; None when not in a git tree or git is
    unavailable. Never raises, never hangs (2s timeout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=2,
        )
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        return None


def _base_provenance() -> dict:
    return {
        "tvboptim": _tvboptim_version(),
        "parrot_neuro": _PARROT_NEURO_VERSION,
        "git_commit": _git_commit(),
        "date": date.today().isoformat(),
    }


def _fixed_params(spec: ModelSpec, learnable: Iterable) -> dict[str, float]:
    """The rest of the spec's defaults, minus whatever ``learnable`` covers and
    minus the structural params -- the generic form of STEP1-SPEC.md's
    completeness invariant. ``learnable`` is anything with ``.name``/``.location``
    (a `Bound` tuple, a `LearnableParam` tuple, or a `_NameLoc` list)."""
    used = {(lp.location, lp.name) for lp in learnable}
    fixed = {
        k: v for k, v in spec.default_params.items()
        if k not in spec.structural_params and ("dynamics", k) not in used
    }
    fixed.update(
        (k, v) for k, v in spec.coupling_defaults.items() if ("coupling", k) not in used
    )
    return fixed


def _sanitize_fit_config(cfg_dict: Mapping[str, Any]) -> dict:
    """Rewrite the two path-valued entries of a verbatim ``BoldFitConfig.to_dict()``
    dump so the sidecar never contains an absolute path (STEP1-SPEC.md's
    "Paths in the sidecar" decision). ``subject.bids_root`` -> dataset name only
    (stored as ``subject.dataset``); ``output_dir`` -> relative to the bids root
    when it is under it, else its basename. Everything else copied verbatim.
    """
    d = dict(cfg_dict)
    subj = dict(d.get("subject") or {})
    bids_root = subj.pop("bids_root", None)
    if bids_root is not None:
        subj["dataset"] = Path(bids_root).name
    d["subject"] = subj

    output_dir = d.get("output_dir")
    if output_dir is not None:
        if bids_root is not None:
            try:
                rel = Path(output_dir).resolve().relative_to(Path(bids_root).resolve())
                d["output_dir"] = str(rel)
            except ValueError:
                d["output_dir"] = Path(output_dir).name
        else:
            d["output_dir"] = Path(output_dir).name
    return d


def _relative_final_loss(history: Iterable[float]) -> Optional[float]:
    """``history[-1] / history[0]``, or None if empty / no valid baseline. Same
    definition as `origin/eeg-bold-fit`'s `train.relative_final_loss` -- duplicated
    (not imported) because that module is jax-heavy; this is 2 lines of arithmetic."""
    h = list(history)
    if not h or not h[0]:
        return None
    return h[-1] / h[0]


def _template_data_dir() -> Path:
    override = os.environ.get("PARROT_TEMPLATE_DATA_DIR")
    if override:
        return Path(override)
    # src/parrot_neuro/simulation/params.py -> repo root is 3 parents up.
    return Path(__file__).resolve().parents[3] / "template_data"


# --- the format --------------------------------------------------------------
@dataclass(frozen=True)
class NMMParams:
    """A neural-mass parameter set on the connectome node axis (M).

    Immutable in spirit (frozen dataclass): every mutator (``override``) returns
    a new object. ``values`` holds one ``(M,)`` array per dynamics parameter and
    ``(1,)`` per coupling parameter; NaN marks a node with no value. See
    STEP1-SPEC.md for the full format decisions this class implements.
    """

    model: str
    model_version: int
    atlas: int
    n_nodes: int  # M, the connectome axis
    values: dict[str, np.ndarray]
    bounds: dict[str, tuple[float, float]]
    locations: dict[str, str]           # "dynamics" | "coupling"
    fixed_params: dict[str, float]
    structural_params: tuple[str, ...]
    fit_config: Optional[dict]
    provenance: dict
    node_space: str = "connectome"

    def __post_init__(self):
        if set(self.values) != set(self.bounds) or set(self.values) != set(self.locations):
            raise ValueError("values/bounds/locations must share the same parameter names")
        for name, arr in self.values.items():
            expected = (1,) if self.locations[name] == "coupling" else (self.n_nodes,)
            if arr.shape != expected:
                raise ValueError(f"{name}: shape {arr.shape} != expected {expected}")

    # --- constructors ---------------------------------------------------
    @classmethod
    def defaults(cls, model: str, atlas: int, n_nodes: int) -> "NMMParams":
        """Homogeneous parameters from the model's own defaults -- no NaN anywhere."""
        spec = get_spec(model)
        values: dict[str, np.ndarray] = {}
        bounds: dict[str, tuple[float, float]] = {}
        locations: dict[str, str] = {}
        for b in spec.learnable:
            shape = (1,) if b.location == "coupling" else (n_nodes,)
            values[b.name] = np.full(shape, spec.resolved_init(b), dtype=np.float64)
            bounds[b.name] = (b.low, b.high)
            locations[b.name] = b.location
        return cls(
            model=spec.name, model_version=spec.version, atlas=atlas, n_nodes=n_nodes,
            values=values, bounds=bounds, locations=locations,
            fixed_params=_fixed_params(spec, spec.learnable),
            structural_params=spec.structural_params,
            fit_config=None, provenance={"kind": "defaults", **_base_provenance()},
        )

    @classmethod
    def load(cls, path: str | Path) -> "NMMParams":
        """Load a saved parameter set. Portable: works from any CWD and after the
        npz/json pair is moved, since both are resolved from ``path`` alone."""
        stem = _stem(path)
        npz_path, json_path = Path(f"{stem}.npz"), Path(f"{stem}.json")
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        with np.load(npz_path) as z:
            values = {k: np.array(z[k], dtype=np.float64) for k in z.files}
        sidecar = json.loads(json_path.read_text())
        bounds = {k: (v["low"], v["high"]) for k, v in sidecar["params"].items()}
        locations = {k: v["location"] for k, v in sidecar["params"].items()}
        return cls(
            model=sidecar["model"], model_version=sidecar["model_version"],
            atlas=sidecar["atlas"], n_nodes=sidecar["n_nodes"], node_space=sidecar["node_space"],
            values=values, bounds=bounds, locations=locations,
            fixed_params=sidecar["fixed_params"],
            structural_params=tuple(sidecar["structural_params"]),
            fit_config=sidecar["fit_config"], provenance=sidecar["provenance"],
        )

    @classmethod
    def from_fit(
        cls,
        values_optim_axis: Mapping[str, np.ndarray],
        cfg: Any,
        *,
        subject: Optional["Subject"] = None,
        loss_history_eeg: Iterable[float] = (),
        loss_history_bold: Iterable[float] = (),
        model: str = "heterogeneous_jr_wc",
    ) -> "NMMParams":
        """Build an ``NMMParams`` from a finished fit, on the K (optim) axis.

        ``values_optim_axis`` is the CALLER's own
        ``train.extract_learnable_values(fit_result.diff_params, cfg.learnable_params)``
        -- the jax-dependent extraction happens on the caller's side; this
        function only ever handles plain numpy arrays (STEP1-SPEC.md's explicit
        direction). ``cfg`` is duck-typed against ``BoldFitConfig``: read
        ``to_dict()``, ``learnable_params``, ``subject``, ``atlas``, ``fmri_task``,
        ``num_epochs``, ``optimize``. Loss histories are plain floats (no jax
        needed either) -- pass ``fit_result.loss_history_eeg`` /
        ``.loss_history_bold`` directly.
        """
        spec = get_spec(model)
        subject = subject if subject is not None else cfg.subject
        nodes = subject.load.fmri_nodes(cfg.atlas, cfg.fmri_task)
        n_nodes = len(nodes.keep)

        loc_by_name = {lp.name: lp.location for lp in cfg.learnable_params}
        bound_by_name = {lp.name: (lp.low, lp.high) for lp in cfg.learnable_params}
        missing = set(values_optim_axis) - set(loc_by_name)
        if missing:
            raise ValueError(f"values for {sorted(missing)} not described in cfg.learnable_params")

        values: dict[str, np.ndarray] = {}
        bounds: dict[str, tuple[float, float]] = {}
        locations: dict[str, str] = {}
        for name, arr in values_optim_axis.items():
            loc = loc_by_name[name]
            locations[name] = loc
            bounds[name] = bound_by_name[name]
            arr = np.asarray(arr, dtype=np.float64)
            if loc == "coupling":
                values[name] = arr.reshape(1)
            else:
                full = np.full(n_nodes, np.nan, dtype=np.float64)
                full[nodes.to_conn] = arr
                values[name] = full

        learnable_used = [_NameLoc(n, loc_by_name[n]) for n in values_optim_axis]
        loss_eeg = list(loss_history_eeg)
        loss_bold = list(loss_history_bold)
        epochs_run = max(len(loss_eeg), len(loss_bold))
        num_epochs = getattr(cfg, "num_epochs", epochs_run)

        provenance = {
            "kind": "subject_fit",
            "subject": subject.subject,
            "dataset": Path(subject.bids_root).name,
            "group": None,
            "optimize": getattr(cfg, "optimize", None),
            "epochs_run": epochs_run,
            "stopped_early": epochs_run < num_epochs,
            "final_loss_eeg": float(loss_eeg[-1]) if loss_eeg else None,
            "final_loss_bold": float(loss_bold[-1]) if loss_bold else None,
            "eeg_loss_ratio": _relative_final_loss(loss_eeg),
            "bold_loss_ratio": _relative_final_loss(loss_bold),
            **_base_provenance(),
        }
        return cls(
            model=spec.name, model_version=spec.version, atlas=cfg.atlas, n_nodes=n_nodes,
            values=values, bounds=bounds, locations=locations,
            fixed_params=_fixed_params(spec, learnable_used),
            structural_params=spec.structural_params,
            fit_config=_sanitize_fit_config(cfg.to_dict()), provenance=provenance,
        )

    @classmethod
    def template(cls, model: str, atlas: int, group: Optional[str] = None) -> Optional["NMMParams"]:
        """Group-level parameter template from ``template_data/nmm_params/``.

        That directory does not exist yet -- it can't, until a cohort has been
        fitted (plan Phase 5) -- so this is a clean no-op: returns None, and
        ``for_subject``'s chain falls through to ``defaults()``. No naming
        convention has been decided for the (not-yet-existing) template files,
        so the "directory exists" branch deliberately raises rather than
        guessing a format.
        """
        d = _template_data_dir() / "nmm_params"
        if not d.is_dir():
            return None
        raise NotImplementedError(
            f"{d} exists but the template file format/naming isn't implemented yet "
            "(plan Phase 5) -- NMMParams.template() only knows how to report its absence."
        )

    @classmethod
    def for_subject(
        cls,
        subject: "Subject",
        atlas: int,
        *,
        model: str = "heterogeneous_jr_wc",
        fit: Optional[str] = None,
        group: Optional[str] = None,
        task: str = "rest",
    ) -> "NMMParams":
        """The resolver: subject_fit -> template -> defaults, merged per node.

        Nodes still NaN after a step fall through to the next; ``provenance``
        records which source(s) actually contributed (``kind``/``chain``) and how
        many entries each filled per parameter (``coverage``).
        """
        spec = get_spec(model)
        n_nodes = int(subject.load.weights(atlas).shape[0])

        sources: list[tuple[str, NMMParams]] = []
        fit_params = cls._discover_subject_fit(subject, atlas, fit)
        if fit_params is not None:
            sources.append(("subject_fit", fit_params))
        tmpl = cls.template(model, atlas, group=group)
        if tmpl is not None:
            sources.append(("template", tmpl))
        sources.append(("defaults", cls.defaults(model, atlas, n_nodes)))

        return cls._merge_chain(sources, spec=spec, atlas=atlas, n_nodes=n_nodes, subject=subject)

    @classmethod
    def from_legacy(
        cls,
        npz_path: str | Path,
        subject: Optional["Subject"] = None,
        *,
        model: str = "heterogeneous_jr_wc",
    ) -> "NMMParams":
        """Migration shim: reads a pre-``NMMParams`` ``optimized_params.npz`` +
        sibling ``config.json`` (the format written by
        ``origin/eeg-bold-fit:examples/eeg_bold_fit_cli.py`` +
        ``BoldFitConfig.save()``) and produces an ``NMMParams``.
        """
        npz_path = Path(npz_path)
        config_path = npz_path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"expected a sibling config.json next to {npz_path}, found none")
        cfg = json.loads(config_path.read_text())

        with np.load(npz_path) as z:
            raw = {k: np.array(z[k]) for k in z.files}
        k_values = {k: v for k, v in raw.items() if k not in _LEGACY_BOOKKEEPING_KEYS}

        learnable_cfg = cfg.get("learnable_params")
        if not learnable_cfg:
            raise ValueError(f"{config_path}: no 'learnable_params' to source parameter locations from")
        loc_by_name = {lp["name"]: lp["location"] for lp in learnable_cfg}
        bound_by_name = {lp["name"]: (lp["low"], lp["high"]) for lp in learnable_cfg}
        missing = set(k_values) - set(loc_by_name)
        if missing:
            raise ValueError(f"{npz_path}: {sorted(missing)} not described in {config_path}'s learnable_params")

        atlas = cfg.get("atlas")
        if atlas is None:
            raise ValueError(f"{config_path}: missing 'atlas'")
        task = cfg.get("fmri_task", "rest")

        if subject is None:
            subj_info = cfg.get("subject") or {}
            if "bids_root" not in subj_info or "subject" not in subj_info:
                raise ValueError(
                    f"no subject given and {config_path} has no reconstructable subject "
                    "info -- pass subject= explicitly"
                )
            from ..subject import Subject  # lazy: keep this module's default import numpy+stdlib-only
            subject = Subject(subj_info["bids_root"], subj_info["subject"])

        nodes = subject.load.fmri_nodes(atlas, task)
        n_nodes = len(nodes.keep)

        spec = get_spec(model)
        values: dict[str, np.ndarray] = {}
        bounds: dict[str, tuple[float, float]] = {}
        locations: dict[str, str] = {}
        for name, arr in k_values.items():
            loc = loc_by_name[name]
            locations[name] = loc
            bounds[name] = bound_by_name[name]
            if loc == "coupling":
                values[name] = np.asarray(arr, dtype=np.float64).reshape(1)
            else:
                full = np.full(n_nodes, np.nan, dtype=np.float64)
                full[nodes.to_conn] = arr
                values[name] = full

        learnable_used = [_NameLoc(n, loc_by_name[n]) for n in k_values]

        loss_eeg = [float(x) for x in np.atleast_1d(raw["loss_eeg"])] if "loss_eeg" in raw else []
        loss_bold = [float(x) for x in np.atleast_1d(raw["loss_bold"])] if "loss_bold" in raw else []
        epochs_run = max(len(loss_eeg), len(loss_bold))
        num_epochs = cfg.get("num_epochs", epochs_run)

        def _ratio(key: str) -> Optional[float]:
            if key not in raw:
                return None
            v = float(raw[key])
            return None if np.isnan(v) else v

        provenance = {
            "kind": "legacy_fit",
            "subject": subject.subject,
            "dataset": Path(subject.bids_root).name,
            "group": None,
            "optimize": cfg.get("optimize"),
            "epochs_run": epochs_run,
            "stopped_early": epochs_run < num_epochs,
            "final_loss_eeg": loss_eeg[-1] if loss_eeg else None,
            "final_loss_bold": loss_bold[-1] if loss_bold else None,
            "eeg_loss_ratio": _ratio("eeg_loss_ratio"),
            "bold_loss_ratio": _ratio("bold_loss_ratio"),
            **_base_provenance(),
        }
        return cls(
            model=spec.name, model_version=spec.version, atlas=atlas, n_nodes=n_nodes,
            values=values, bounds=bounds, locations=locations,
            fixed_params=_fixed_params(spec, learnable_used),
            structural_params=spec.structural_params,
            fit_config=_sanitize_fit_config(cfg), provenance=provenance,
        )

    # --- internal: the fallback-chain merge -----------------------------
    @staticmethod
    def _discover_subject_fit(subject: "Subject", atlas: int, fit: Optional[str]) -> Optional["NMMParams"]:
        d = subject.path.nmmfit_dir()
        candidates = sorted({
            p.parent.name for p in d.glob(f"*/*_atlas-{atlas}_desc-nmmparams.npz")
        }) if d.is_dir() else []
        if not candidates:
            return None
        if fit is None:
            if len(candidates) > 1:
                raise ValueError(
                    f"multiple nmm fits found for {subject.subj} atlas={atlas}: {candidates}; "
                    "pass fit=<name> to select one"
                )
            name = candidates[0]
        else:
            if fit not in candidates:
                raise ValueError(f"no nmm fit {fit!r} for {subject.subj} atlas={atlas}; found: {candidates}")
            name = fit
        return NMMParams.load(subject.path.nmm_params(name, atlas))

    @classmethod
    def _merge_chain(
        cls, sources: list[tuple[str, "NMMParams"]], *, spec: ModelSpec, atlas: int,
        n_nodes: int, subject: "Subject",
    ) -> "NMMParams":
        bound_by_name = {b.name: b for b in spec.learnable}
        values: dict[str, np.ndarray] = {}
        coverage: dict[str, dict[str, int]] = {}
        contributed: set[str] = set()

        for name, b in bound_by_name.items():
            shape = (1,) if b.location == "coupling" else (n_nodes,)
            acc = np.full(shape, np.nan, dtype=np.float64)
            for kind, p in sources:
                if name not in p.values or p.values[name].shape != shape:
                    continue
                arr = p.values[name]
                fill = np.isnan(acc) & np.isfinite(arr)
                n_filled = int(np.sum(fill))
                if n_filled:
                    acc[fill] = arr[fill]
                    coverage.setdefault(name, {})[kind] = n_filled
                    contributed.add(kind)
                if not np.any(np.isnan(acc)):
                    break
            values[name] = acc

        # preserve true priority order, not per-name discovery order
        chain = [kind for kind, _ in sources if kind in contributed]
        kind = chain[0] if len(chain) == 1 else "mixed"

        provenance = {
            "kind": kind,
            "chain": chain,
            "coverage": coverage,
            "subject": subject.subject,
            "dataset": Path(subject.bids_root).name,
            "group": None,
            **_base_provenance(),
        }
        return cls(
            model=spec.name, model_version=spec.version, atlas=atlas, n_nodes=n_nodes,
            values=values,
            bounds={n: (b.low, b.high) for n, b in bound_by_name.items()},
            locations={n: b.location for n, b in bound_by_name.items()},
            fixed_params=_fixed_params(spec, spec.learnable),
            structural_params=spec.structural_params,
            fit_config=None, provenance=provenance,
        )

    # --- instance API -----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write ``<stem>.npz`` + ``<stem>.json``. ``path`` may be either; no
        directory magic beyond creating the parent."""
        extra = set(self.provenance) - _PROVENANCE_ALLOWED_KEYS
        if extra:
            raise ValueError(
                f"provenance has disallowed key(s) {sorted(extra)}; "
                f"allowed: {sorted(_PROVENANCE_ALLOWED_KEYS)}"
            )
        stem = _stem(path)
        npz_path, json_path = Path(f"{stem}.npz"), Path(f"{stem}.json")
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in self.values.items()}
        np.savez(npz_path, **arrays)

        sidecar = {
            "model": self.model, "model_version": self.model_version,
            "atlas": self.atlas, "node_space": self.node_space, "n_nodes": self.n_nodes,
            "params": {
                name: {"shape": list(arr.shape), "low": self.bounds[name][0],
                       "high": self.bounds[name][1], "location": self.locations[name]}
                for name, arr in arrays.items()
            },
            "fixed_params": self.fixed_params,
            "structural_params": list(self.structural_params),
            "fit_config": self.fit_config,
            "provenance": self.provenance,
        }
        json_path.write_text(json.dumps(sidecar, indent=2, default=str) + "\n")
        return npz_path

    def to_optim_axis(
        self, subject: "Subject", task: str = "rest", allow_nan: bool = False
    ) -> dict[str, np.ndarray]:
        """Gather each ``(M,)`` value onto the subject's ``(K,)`` optim axis;
        ``(1,)`` coupling values pass through unchanged. Raises if a kept node's
        value is NaN unless ``allow_nan=True``."""
        nodes = subject.load.fmri_nodes(self.atlas, task)
        out: dict[str, np.ndarray] = {}
        for name, arr in self.values.items():
            if self.locations[name] == "coupling":
                out[name] = np.array(arr, copy=True)
                continue
            gathered = arr[nodes.to_conn]
            if not allow_nan and not np.all(np.isfinite(gathered)):
                bad = nodes.to_conn[np.flatnonzero(~np.isfinite(gathered))]
                raise ValueError(
                    f"{name}: {len(bad)} kept optim node(s) have no value (NaN) at connectome "
                    f"row(s) {bad.tolist()}; pass allow_nan=True to allow this"
                )
            out[name] = gathered
        return out

    def override(self, **kw: Any) -> "NMMParams":
        """Scalar or ``(M,)``/``(1,)`` array overrides -> a NEW ``NMMParams``."""
        new_values = dict(self.values)
        for name, val in kw.items():
            if name not in self.values:
                raise ValueError(f"unknown parameter {name!r}; known: {sorted(self.values)}")
            target_shape = self.values[name].shape
            arr = np.asarray(val, dtype=np.float64)
            if arr.shape == ():
                arr = np.full(target_shape, float(arr))
            elif arr.shape != target_shape:
                raise ValueError(f"override for {name!r} has shape {arr.shape}, expected {target_shape}")
            new_values[name] = arr
        return replace(self, values=new_values)

    @property
    def coverage(self) -> dict[str, float]:
        """Fraction of finite entries, per parameter."""
        return {name: float(np.mean(np.isfinite(arr))) for name, arr in self.values.items()}

    def describe(self) -> str:
        lines = [
            f"NMMParams(model={self.model!r}, atlas={self.atlas}, n_nodes={self.n_nodes}, "
            f"kind={self.provenance.get('kind')!r})"
        ]
        chain = self.provenance.get("chain")
        if chain:
            lines.append(f"  chain: {' -> '.join(chain)}")
        for name, frac in sorted(self.coverage.items()):
            lines.append(f"  {name}: coverage={frac:.2f}")
        return "\n".join(lines)


def _stem(path: str | Path) -> str:
    """``path`` may be ``<stem>.npz`` or a bare ``<stem>``; strip a literal
    ``.npz`` suffix by string, never via ``Path.with_suffix`` (which would
    mangle a stem containing its own dots, e.g. a leadfield-style key)."""
    s = str(path)
    return s[: -len(".npz")] if s.endswith(".npz") else s
