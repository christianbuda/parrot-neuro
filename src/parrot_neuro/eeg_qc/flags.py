"""Turn :class:`~.metrics.ChannelMetrics` into pass/warn/fail calls per channel,
plus the two "missing electrode" checks: channels the recording is missing
relative to an expected montage, and recorded channels with no matching
position in the subject's own electrode montage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .checks import FAIL, PASS, WARN, worst
from .metrics import ChannelMetrics, robust_z


@dataclass
class Thresholds:
    """Warn/fail cutoffs. Amplitude/kurtosis/HF-noise/segment-consistency are
    judged relative to the other channels of the *same* recording (a robust
    z-score across channels — ``robust_z``), since "normal" amplitude varies
    a lot between subjects/tasks/preprocessing. Line-noise and neighbor
    correlation are judged on an absolute scale instead, since those have a
    physically meaningful unit (a ratio to a local baseline; a correlation
    coefficient) that doesn't need renormalizing per recording."""

    flatline_fraction_warn: float = 0.02
    flatline_fraction_fail: float = 0.10
    amplitude_z_warn: float = 3.0
    amplitude_z_fail: float = 5.0
    kurtosis_z_warn: float = 3.0
    kurtosis_z_fail: float = 5.0
    hf_noise_z_warn: float = 3.0
    hf_noise_z_fail: float = 5.0
    line_noise_ratio_warn: float = 2.5
    line_noise_ratio_fail: float = 4.0
    neighbor_corr_warn: float = 0.6
    neighbor_corr_fail: float = 0.4
    segment_cv_z_warn: float = 3.0
    segment_cv_z_fail: float = 5.0


@dataclass
class ChannelFlag:
    name: str
    status: str
    reasons: list[str] = field(default_factory=list)


def _tier(value: float, warn: float, fail: float, high_is_bad: bool = True) -> str:
    if high_is_bad:
        if value >= fail:
            return FAIL
        if value >= warn:
            return WARN
    else:
        if value <= fail:
            return FAIL
        if value <= warn:
            return WARN
    return PASS


def flag_channels(
    metrics: ChannelMetrics,
    positions: dict[str, np.ndarray] | None = None,
    thresholds: Thresholds | None = None,
) -> list[ChannelFlag]:
    th = thresholds or Thresholds()
    amp_z = np.abs(robust_z(metrics.rms))
    kurt_z = robust_z(metrics.kurtosis)
    hf_z = robust_z(metrics.hf_noise_ratio)
    cv_z = robust_z(metrics.segment_std_cv)
    median_rms = np.median(metrics.rms)

    flags = []
    for i, name in enumerate(metrics.channel_names):
        statuses, reasons = [], []

        s = _tier(metrics.flatline_fraction[i], th.flatline_fraction_warn, th.flatline_fraction_fail)
        statuses.append(s)
        if s != PASS:
            reasons.append(f"flatline ({metrics.flatline_fraction[i] * 100:.1f}% near-zero samples)")

        s = _tier(amp_z[i], th.amplitude_z_warn, th.amplitude_z_fail)
        statuses.append(s)
        if s != PASS:
            direction = "high" if metrics.rms[i] > median_rms else "low"
            reasons.append(f"amplitude outlier ({direction}, z={amp_z[i]:.1f})")

        s = _tier(kurt_z[i], th.kurtosis_z_warn, th.kurtosis_z_fail)
        statuses.append(s)
        if s != PASS:
            reasons.append(f"high kurtosis / spiky (z={kurt_z[i]:.1f})")

        s = _tier(hf_z[i], th.hf_noise_z_warn, th.hf_noise_z_fail)
        statuses.append(s)
        if s != PASS:
            reasons.append(f"excess high-frequency power (z={hf_z[i]:.1f})")

        s = _tier(metrics.line_noise_ratio[i], th.line_noise_ratio_warn, th.line_noise_ratio_fail)
        statuses.append(s)
        if s != PASS:
            reasons.append(f"line-noise peak ({metrics.line_noise_ratio[i]:.1f}x local baseline)")

        s = _tier(metrics.neighbor_corr[i], th.neighbor_corr_warn, th.neighbor_corr_fail, high_is_bad=False)
        statuses.append(s)
        if s != PASS:
            reasons.append(f"poor correlation with nearby electrodes (r={metrics.neighbor_corr[i]:.2f})")

        s = _tier(cv_z[i], th.segment_cv_z_warn, th.segment_cv_z_fail)
        statuses.append(s)
        if s != PASS:
            reasons.append(f"inconsistent amplitude across segments (z={cv_z[i]:.1f})")

        if positions is not None and name not in positions:
            statuses.append(FAIL)
            reasons.append("no electrode position on record (name not found in montage)")

        flags.append(ChannelFlag(name, worst(statuses), reasons))
    return flags


def missing_channels(recorded_names, expected_names) -> list[str]:
    """Names in ``expected_names`` absent from ``recorded_names`` -- electrodes
    that should have been recorded (e.g. the montage used in a companion task,
    or a dataset-wide reference cap layout) but weren't."""
    recorded = set(recorded_names)
    return [n for n in expected_names if n not in recorded]
