"""Shared loading + metrics for the article scripts in ``examples/article/``.

Everything here reads saved fit outputs (``eeg_bold_fit_cli.py`` /
``gpu_fit.py`` run folders) and the BIDS derivatives with numpy only -- no
jax, so the CPU analyses never touch a GPU.

A run folder is ``<root>/<subject>_<optimize>[_<schedule>]/`` holding
``config.json``, ``optimized_params.npz``, ``best_params.npz``,
``init_params.npz``, ``diagnostics_metrics.json`` and
``simulation_results.npz`` (the arrays the post-fit figures are drawn from,
simulated from the best-epoch params, plus the same at the initial params).
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))

from parrot_neuro import Subject  # noqa: E402

DEFAULT_ROOT = Path("new_leadfields/2/atlas-100")
DEFAULT_BIDS = Path("/srv/nfs-data/sisko/christian/parrot_LEMON")

# reference palette (dataviz skill, references/palette.md): categorical slots
# in fixed order, the same young/old pair and ramps as the population report
C_FIT, C_INIT, C_SC, C_ALT = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
YOUNG_C, OLD_C, NEUTRAL_C = "#2a78d6", "#eb6834", "#a9b3ac"
INK, INK_2 = "#0b0b0b", "#52514e"

ALPHA_BAND = (8.0, 12.0)        # relative alpha power / topography
APF_BAND = (7.0, 14.0)          # where the alpha peak is searched
APERIODIC_BAND = (2.0, 40.0)    # 1/f exponent fit range (alpha band excluded)
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}


def posterior(ch: str) -> bool:
    """Occipital / parieto-occipital / parietal channels (alpha ROI)."""
    c = ch.upper()
    return c.startswith(("O", "PO", "P")) and not c.startswith("FP")


# ---------------------------------------------------------------------------
# runs

@dataclass
class Run:
    path: Path
    cfg: dict
    diag: dict = field(default_factory=dict)

    @property
    def sid(self) -> str:
        return self.cfg["subject"]["subject"]

    @cached_property
    def sim(self):
        return np.load(self.path / "simulation_results.npz")

    @property
    def has_sim(self) -> bool:
        return (self.path / "simulation_results.npz").exists()

    def params(self, which: str = "best") -> dict[str, np.ndarray]:
        """Fitted per-node params (``best`` epoch, the set the diagnostics
        simulate, or ``final``/``init``)."""
        name = {"best": "best_params.npz", "final": "optimized_params.npz", "init": "init_params.npz"}[which]
        f = self.path / name
        if which == "best" and not f.exists():
            f = self.path / "optimized_params.npz"
        z = np.load(f)
        return {lp["name"]: np.asarray(z[lp["name"]], float) for lp in self.cfg["learnable_params"]
                if lp["name"] in z.files}

    @cached_property
    def losses(self) -> dict[str, float]:
        """Mean of the last 10 epochs of the raw (unweighted) training losses."""
        z = np.load(self.path / "optimized_params.npz")
        out = {}
        for k in ("loss_eeg", "loss_bold"):
            v = np.asarray(z[k], float) if k in z.files else np.array([])
            v = v[np.isfinite(v) & (v > 0)]
            out[k] = float(np.mean(v[-10:])) if len(v) else np.nan
        return out


def load_runs(root: Path, need_sim: bool = True) -> list[Run]:
    runs = []
    for d in sorted(Path(root).iterdir()):
        if not (d / "config.json").exists() or not (d / "optimized_params.npz").exists():
            continue
        if need_sim and not (d / "simulation_results.npz").exists():
            continue
        diag = json.loads((d / "diagnostics_metrics.json").read_text()) if (d / "diagnostics_metrics.json").exists() else {}
        runs.append(Run(d, json.loads((d / "config.json").read_text()), diag))
    if not runs:
        raise SystemExit(f"no finished runs under {root}")
    return runs


def participants(bids: Path, age_split: float = 45.0) -> pd.DataFrame:
    p = pd.read_csv(bids / "participants.tsv", sep="\t")
    p["subject"] = p.participant_id.str.replace("sub-", "", regex=False)
    p["old"] = p.age >= age_split
    return p


# ---------------------------------------------------------------------------
# spectra (numpy twins of optimization.signal.compute_psd / smooth_ts)

def psd(x: np.ndarray) -> np.ndarray:
    """(channels, time) -> (channels, freqs): Hann-windowed |rfft|^2 / T,
    exactly ``signal.compute_psd``."""
    T = x.shape[-1]
    return np.abs(np.fft.rfft(x * np.hanning(T), axis=-1)) ** 2 / T


def smooth(P: np.ndarray, w: int = 5) -> np.ndarray:
    """Moving average along frequency, ``mode="same"`` -- ``signal.smooth_ts``."""
    k = np.ones(w) / w
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), -1, np.atleast_2d(P))


def _band(freqs, lo, hi):
    return (freqs >= lo) & (freqs <= hi)


def aperiodic_fit(freqs: np.ndarray, P: np.ndarray, band=APERIODIC_BAND, exclude=APF_BAND):
    """Line in log-log space over ``band`` minus ``exclude`` -> (exponent,
    intercept) per row; exponent > 0 = steeper 1/f."""
    m = _band(freqs, *band) & ~_band(freqs, *exclude) & (freqs > 0)
    lf = np.log10(freqs[m])
    lp = np.log10(np.atleast_2d(P)[:, m])
    A = np.column_stack([lf, np.ones_like(lf)])
    coef = np.linalg.lstsq(A, lp.T, rcond=None)[0]
    return -coef[0], coef[1]


def alpha_peak(freqs: np.ndarray, P: np.ndarray, fit_band=(1.0, 15.0)) -> np.ndarray:
    """Individual alpha frequency per row: centre of gravity of the
    aperiodic-removed (log) spectrum's positive part within ``APF_BAND``.
    The aperiodic line is fitted inside the band the model was fitted on
    (``fit_band``), so model and data are treated the same way."""
    P = np.atleast_2d(P)
    lo = max(fit_band[0], 1.0)
    expo, icpt = aperiodic_fit(freqs, P, band=(lo, fit_band[1]), exclude=APF_BAND)
    m = _band(freqs, *APF_BAND)
    resid = np.log10(P[:, m]) - (icpt[:, None] - expo[:, None] * np.log10(freqs[m]))
    w = np.clip(resid, 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w * freqs[m]).sum(1) / w.sum(1)


def eeg_metrics(sim: np.ndarray, emp: np.ndarray, freqs: np.ndarray, names: list[str],
                fmin: float = 1.0, fmax: float = 15.0) -> dict:
    """Interpretable sim-vs-empirical EEG agreement for one subject
    (channels x freqs PSDs on the same channel order), all on the smoothed
    spectra the loss compares:

    - ``spec_r``: median over channels of the Pearson r between log10 PSDs
      inside the fitted band (spectral shape, per channel)
    - ``apf_sim``/``apf_emp``: alpha peak frequency on the posterior ROI mean
    - ``alpha_topo_r``: Pearson r across channels of relative alpha power
      (alpha / [fmin, fmax] power, per channel)
    - ``exp_sim``/``exp_emp``: aperiodic exponent (2-40 Hz, alpha excluded);
      above ``fmax`` the model was never fitted, so the simulated one is a
      prediction
    - ``rel_<band>_sim``/``_emp``: posterior relative band power
    """
    S, E = smooth(sim), smooth(emp)
    b = _band(freqs, fmin, fmax)
    r = [stats.pearsonr(np.log10(S[c, b]), np.log10(E[c, b]))[0] for c in range(len(S))]
    roi = np.array([posterior(n) for n in names])
    roi = roi if roi.any() else np.ones(len(names), bool)
    a = _band(freqs, *ALPHA_BAND)
    rel = lambda P: P[:, a].sum(1) / P[:, b].sum(1)  # noqa: E731
    out = {
        "spec_r": float(np.nanmedian(r)),
        "spec_r_roi": float(np.nanmedian(np.asarray(r)[roi])),
        "apf_sim": float(alpha_peak(freqs, S[roi].mean(0), (fmin, fmax))[0]),
        "apf_emp": float(alpha_peak(freqs, E[roi].mean(0), (fmin, fmax))[0]),
        "alpha_topo_r": float(stats.pearsonr(rel(S), rel(E))[0]),
        "exp_sim": float(aperiodic_fit(freqs, S[roi].mean(0))[0][0]),
        "exp_emp": float(aperiodic_fit(freqs, E[roi].mean(0))[0][0]),
    }
    for k, (lo, hi) in BANDS.items():
        m = _band(freqs, lo, hi)
        for tag, P in (("sim", S), ("emp", E)):
            out[f"rel_{k}_{tag}"] = float((P[roi][:, m].sum(1) / P[roi][:, b].sum(1)).mean())
    return out


def fit_metrics(run: "Run", subj: Subject) -> dict:
    """Both modalities' agreement for one run (fitted params), from its
    ``simulation_results.npz`` -- every fit's diagnostics simulate EEG and
    BOLD, whichever loss(es) it optimized."""
    z = run.sim
    m = eeg_metrics(z["eeg_psd_sim"], z["eeg_psd_target"], z["eeg_freqs"], channel_names(run, subj),
                    run.cfg["fmin"], run.cfg["fmax"])
    out = {"subject": run.sid, "spec_r": m["spec_r"], "alpha_topo_r": m["alpha_topo_r"],
           "apf_absdiff": abs(m["apf_sim"] - m["apf_emp"]), "fc_r": fc_r(z["fc_sim"], z["fc_emp"])}
    if "fcd_sim" in z.files:
        out["fcd_ks"] = fcd_ks(z["fcd_sim"], z["fcd_emp"], int(run.cfg.get("dfc_kmin", 1)))
    out.update({k: v for k, v in run.losses.items()})
    return out


def spectral_similarity(sim: np.ndarray, emp: np.ndarray, freqs: np.ndarray,
                        fmin: float = 1.0, fmax: float = 15.0) -> float:
    """Single number for cross-subject matrices: median over channels of the
    log-PSD correlation in the fitted band (``spec_r``)."""
    b = _band(freqs, fmin, fmax)
    S, E = np.log10(smooth(sim)[:, b]), np.log10(smooth(emp)[:, b])
    S = (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
    E = (E - E.mean(1, keepdims=True)) / E.std(1, keepdims=True)
    return float(np.nanmedian((S * E).mean(1)))


# ---------------------------------------------------------------------------
# empirical EEG chunks (numpy twin of data.load_subject_eeg)

def eeg_chunks(subj: Subject, task: str, chunk_length: int) -> tuple[list[np.ndarray], list[str]]:
    """The subject's EEG cut into the same ``chunk_length`` chunks
    ``data.SingleSubjectDataset`` makes, in the same order, plus the channel
    names (recording order = the PSD rows the fit saved)."""
    z = subj.load.eeg(task)
    meta = json.loads(subj.path.eeg(task).with_suffix(".json").read_text())
    chunks = []
    for k in sorted(z.files):
        rec = z[k]
        n = rec.shape[1] // chunk_length
        if n:
            chunks += np.split(rec[:, : n * chunk_length], n, axis=1)
    return chunks, [str(c) for c in meta["channel_names"]]


def channel_names(run: Run, subj: Subject) -> list[str]:
    el = np.loadtxt(subj.path.electrodes_csv(), delimiter=",", usecols=0, dtype=str)
    return [str(el[i]).strip() for i in run.sim["eeg_channel_indices"]]


def holdout_blocks(n_chunks: int, block: int) -> np.ndarray:
    """Half label (0 = A, 1 = B) of each chunk: alternating blocks of
    ``block`` consecutive chunks, so both halves span the whole recording
    while neighbouring (autocorrelated) chunks mostly stay together."""
    return (np.arange(n_chunks) // block) % 2


# ---------------------------------------------------------------------------
# BOLD

def upper(M: np.ndarray, k: int = 1) -> np.ndarray:
    return M[np.triu_indices(M.shape[0], k)]


def fc(X: np.ndarray) -> np.ndarray:
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return np.corrcoef(X, rowvar=False)


def fc_r(A: np.ndarray, B: np.ndarray) -> float:
    a, b = upper(A), upper(B)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def fcd_ks(fcd_sim: np.ndarray, fcd_emp: np.ndarray, k_min: int = 1) -> float:
    """KS distance between the FCD value distributions (window-count agnostic)."""
    return float(stats.ks_2samp(upper(fcd_sim, k_min), upper(fcd_emp, k_min)).statistic)


def bandpass_bold(X: np.ndarray, tr_ms: float, low: float = 0.01, high: float = 0.1, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth, as ``connectivity.filter_sim_bold`` (for
    simulated BOLD only -- the empirical series are already filtered)."""
    from scipy.signal import butter, filtfilt
    fs = 1000.0 / tr_ms
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, X, axis=0)


def to_conn(subj: Subject, cfg: dict) -> np.ndarray:
    return np.asarray(subj.load.fmri_nodes(cfg["atlas"], cfg["fmri_task"]).to_conn)


def scatter_matrix(M: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    """Subject-node (k, k) matrix onto the shared connectome axis (n, n), NaN elsewhere."""
    out = np.full((n, n), np.nan)
    out[np.ix_(idx, idx)] = M
    return out


# ---------------------------------------------------------------------------
# fMRI motion

def mean_fd(subj: Subject, task: str = "rest", rot_cols=(6, 7, 8), trans_cols=(9, 10, 11),
            radius_mm: float = 50.0) -> float:
    """Power framewise displacement from the staged LEMON ``confounds.txt``.
    Columns (0-based) 6-8 hold rotations (rad), 9-11 translations (mm) --
    columns 12-17 are exactly their first differences, 0-5 aCompCor and
    18-20 polynomial trends (checked on the staged files)."""
    f = subj.path.fmri_bold(task).with_name(f"{subj.subj}_task-{task}_confounds.txt")
    if not f.exists():
        return np.nan
    X = np.loadtxt(f)
    d = np.abs(np.diff(X, axis=0))
    return float((radius_mm * d[:, list(rot_cols)].sum(1) + d[:, list(trans_cols)].sum(1)).mean())


def head_radius_mm(subj: Subject) -> float:
    """Mean electrode distance from the montage centroid (head size)."""
    xyz = np.loadtxt(subj.path.electrodes_csv(), delimiter=",", usecols=(1, 2, 3))
    return float(np.linalg.norm(xyz - xyz.mean(0), axis=1).mean())


# ---------------------------------------------------------------------------
# output

class Out:
    """``<out>/figures``, ``<out>/tables`` and a ``numbers.json`` of the
    headline values the text quotes (merged across scripts)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "figures").mkdir(parents=True, exist_ok=True)
        (self.root / "tables").mkdir(parents=True, exist_ok=True)

    def fig(self, fig, name: str) -> None:
        for ext in ("png", "pdf"):
            fig.savefig(self.root / "figures" / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  figures/{name}.png")

    def table(self, df: pd.DataFrame, name: str, index: bool = False) -> None:
        df.to_csv(self.root / "tables" / f"{name}.csv", index=index)
        print(f"  tables/{name}.csv")

    def numbers(self, section: str, values: dict) -> None:
        f = self.root / "numbers.json"
        d = json.loads(f.read_text()) if f.exists() else {}
        d[section] = _clean(values)
        f.write_text(json.dumps(d, indent=2))


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(f"{float(o):.5g}")
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def summary(x) -> dict:
    """median [IQR] + n of a vector (NaNs dropped)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q1, m, q3 = np.percentile(x, [25, 50, 75])
    return {"median": m, "q1": q1, "q3": q3, "mean": x.mean(), "n": len(x)}


def wilcoxon(a, b) -> dict:
    """Paired Wilcoxon a vs b + rank-biserial effect size (>0: a > b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.wilcoxon(a[ok], b[ok])
    r = (np.sum(d > 0) - np.sum(d < 0)) / max(np.sum(d != 0), 1)
    return {"n": int(ok.sum()), "median_diff": float(np.median(d)), "p": float(res.pvalue),
            "frac_a_gt_b": float(np.mean(d > 0)), "sign_effect": float(r)}


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK_2,
        "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "figure.dpi": 120, "pdf.fonttype": 42,
    })


def panel(ax, letter: str) -> None:
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
