"""Signal-processing and spectral helpers (resampling, PSD, smoothing).

Pure functions, no global state. The JAX-jitted ones take everything they
need as arguments so they can be reused and unit-tested in isolation.
"""
from __future__ import annotations

from fractions import Fraction
from functools import partial

import jax
import jax.numpy as jnp
import scipy.signal


def resample_signals(signals, original_fs, target_fs):
    """Safely resample ``signals`` to an arbitrary target frequency.

    The dimension to resample is the last one. Uses polyphase resampling with
    a rational up/down ratio (denominator capped for a tractable filter).
    """
    ratio_float = target_fs / original_fs
    ratio_fraction = Fraction(ratio_float).limit_denominator(100000)
    up, down = ratio_fraction.numerator, ratio_fraction.denominator

    actual_fs = original_fs * (up / down)
    if abs(actual_fs - target_fs) > 0.1:
        print(
            f"Warning: Target FS {target_fs} approximated to {actual_fs} "
            "to reduce computational burden."
        )
    return scipy.signal.resample_poly(signals, up=up, down=down, axis=-1)


@jax.jit
def compute_psd(signal):
    """Power spectral density over time. Signal shape: (Channels, Time)."""
    signal = jnp.transpose(signal)
    time_steps = signal.shape[0]
    window = jnp.hanning(time_steps)[:, None]
    windowed_signal = signal * window
    fft_out = jnp.fft.rfft(windowed_signal, axis=0)
    psd = jnp.abs(fft_out) ** 2
    return jnp.transpose(psd / time_steps)


@partial(jax.jit, static_argnames=["window_size"])
def smooth_ts(signal, window_size=3):
    """Moving-average filter along the last axis. Shape: (Channels, Time)."""
    kernel = jnp.ones(window_size) / window_size
    smooth_fn = lambda x: jnp.convolve(x, kernel, mode="same")  # noqa: E731
    return jax.vmap(smooth_fn)(signal)


def decouple_spectrum(freqs, psd):
    """Differentiably split a PSD into a 1/f^a slope and periodic residuals.

    Args:
        freqs: 1D array of frequencies (must exclude 0 Hz).
        psd: 1D or 2D array of PSD values (Channels, Freqs).

    Returns:
        alpha: the 1/f exponent (Channels, 1).
        residuals: periodic peaks with the aperiodic baseline removed.
    """
    log_f = jnp.log10(freqs)
    log_psd = jnp.log10(psd)

    # Differentiable OLS in log-log space.
    mu_x = jnp.mean(log_f)
    mu_y = jnp.mean(log_psd, axis=-1, keepdims=True)
    dx = log_f - mu_x
    dy = log_psd - mu_y
    slope = jnp.sum(dx * dy, axis=-1, keepdims=True) / jnp.sum(dx**2)
    intercept = mu_y - slope * mu_x

    aperiodic_baseline = intercept + slope * log_f
    residuals = log_psd - aperiodic_baseline
    alpha = -slope
    return alpha, residuals


def welch_psd(signal, fs=250, nperseg=128, noverlap=64):
    """Welch PSD estimate (segment-averaged periodogram), JAX/vmap-based.

    Signal shape: (Channels, Time) or (Time,). Returns (Channels, nfreqs).
    """
    signal = jnp.atleast_2d(signal)
    channels, time_steps = signal.shape
    step = nperseg - noverlap
    num_segments = (time_steps - noverlap) // step
    starts = jnp.arange(num_segments) * step

    def get_segment(start):
        return jax.lax.dynamic_slice(signal, (0, start), (channels, nperseg))

    segments = jax.vmap(get_segment)(starts)
    segments = segments - jnp.mean(segments, axis=-1, keepdims=True)
    window = jnp.hanning(nperseg)
    windowed_segments = segments * window
    fft_out = jnp.fft.rfft(windowed_segments, axis=-1)
    psd_segments = jnp.abs(fft_out) ** 2
    psd = jnp.mean(psd_segments, axis=0)
    window_sum_squares = jnp.sum(window**2)
    psd = psd / (fs * window_sum_squares)
    # One-sided PSD: double all but DC (and Nyquist when nperseg is even).
    if nperseg % 2 == 0:
        psd = psd.at[:, 1:-1].multiply(2.0)
    else:
        psd = psd.at[:, 1:].multiply(2.0)
    return psd
