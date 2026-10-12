"""
Compare signal quality across all MEG hackathon sessions/runs.

Adapted from Bonsai_ABCD/preprocessing/quality_control/compare_signal_quality.py.
The pickles only contain dF/F (no raw F/Fneu/meanImg), so the metrics here are the
ones that can be computed from dF/F alone:

    noise_sd    frame-to-frame diff MAD (robust to slow drift and transients)
    snr         (99th pct - median) / noise_sd   (dF/F here is not baseline-zeroed)
    skew        skew of each trace (transient-like activity -> positive skew)
    event_frac  fraction of frames > median + event_sigma * noise_sd
    lag1_ac     lag-1 autocorrelation (calcium transients span frames -> high;
                white noise -> ~0)

ROIs whose trace contains NaNs are excluded from these metrics and counted in
frac_nan_rois (some runs have most or all ROIs NaN in one channel).

Channel 2 (activity-independent) gets the same metrics as a control: it should have
low SNR/skew. A high functional-vs-channel-2 correlation flags shared artefacts
(motion, brightness drift) rather than calcium activity.

    python compare_signal_quality.py [--recompute] [--out-dir qc]

Per-run metrics are cached in <out-dir>/signal_quality_runs.csv and only missing
runs are computed on later calls (unless --recompute).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from parser_functions import DATA_DIR, get_F, get_trials, first_reward_frames, list_runs, load_run

PARAMS = {"event_sigma": 3, "snr_threshold": 3, "skew_threshold": 1}

# dataviz reference palette chrome (references/palette.md)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
FUNC_COLOR = "#2a78d6"   # categorical slot 1 (blue)
CHAN2_COLOR = "#eb6834"  # categorical slot 2 (orange)


def trace_metrics(X, event_sigma=3):
    """Per-ROI metrics for a (frames x ROIs) dF/F array, skipping ROIs with NaNs."""
    X = X[:, ~np.isnan(X).any(axis=0)]
    diffs = np.diff(X, axis=0)
    noise_sd = 1.4826 * np.median(np.abs(diffs - np.median(diffs, axis=0)), axis=0) / np.sqrt(2)
    med = np.median(X, axis=0)
    peak = np.percentile(X, 99, axis=0) - med
    return {
        "noise_sd": noise_sd,
        "peak_dff": peak,
        "snr": peak / noise_sd,
        "skew": stats.skew(X, axis=0, nan_policy="omit"),
        "event_frac": np.mean(X > med + event_sigma * noise_sd, axis=0),
        "lag1_ac": column_corr(X[:-1], X[1:]),
    }


def summarise(m, prefix, snr_threshold, skew_threshold):
    if len(m["snr"]) == 0:
        return {}
    return {
        f"{prefix}median_noise_sd": float(np.nanmedian(m["noise_sd"])),
        f"{prefix}median_peak_dff": float(np.nanmedian(m["peak_dff"])),
        f"{prefix}median_snr": float(np.nanmedian(m["snr"])),
        f"{prefix}median_skew": float(np.nanmedian(m["skew"])),
        f"{prefix}median_event_frac": float(np.nanmedian(m["event_frac"])),
        f"{prefix}median_lag1_ac": float(np.nanmedian(m["lag1_ac"])),
        f"{prefix}frac_snr_gt_thresh": float(np.mean(m["snr"] > snr_threshold)),
        f"{prefix}frac_skew_gt_thresh": float(np.mean(m["skew"] > skew_threshold)),
    }


def column_corr(A, B):
    """Pearson correlation between matching columns of A and B."""
    Az = (A - A.mean(0)) / A.std(0)
    Bz = (B - B.mean(0)) / B.std(0)
    return np.mean(Az * Bz, axis=0)


def run_metrics(session, run, roi_matches, params=PARAMS):
    df = load_run(session, run)
    F, t = get_F(df)
    C2, _ = get_F(df, "channel2")
    trials = get_trials(df)

    row = {
        "session": session,
        "run": run,
        "n_frames": len(df),
        "duration_min": (t[-1] - t[0]) / 60,
        "frame_rate_hz": 1 / np.median(np.diff(t)),
        "frac_alignment_valid": float(df["alignment_valid"].mean()),
        "n_rois": F.shape[1],
        "frac_nan_rois": float(np.isnan(F).any(axis=0).mean()),
        "chan2_frac_nan_rois": float(np.isnan(C2).any(axis=0).mean()),
        "n_trials": len(trials),
        "n_rewards": len(first_reward_frames(df)),
    }

    m = roi_matches[roi_matches.session.str.startswith(session + "_") & (roi_matches.run == run)]
    row["n_rois_in_roi_matches"] = len(m)
    row["frac_rois_matched"] = float((m.cluster_id >= 0).mean()) if len(m) else np.nan

    row.update(summarise(trace_metrics(F, params["event_sigma"]), "",
                         params["snr_threshold"], params["skew_threshold"]))
    row.update(summarise(trace_metrics(C2, params["event_sigma"]), "chan2_",
                         params["snr_threshold"], params["skew_threshold"]))
    r = column_corr(F, C2)
    row["median_func_chan2_corr"] = float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan
    return row


def gather_results(out_csv, recompute=False, data_dir=DATA_DIR):
    roi_matches = pd.read_csv(Path(data_dir) / "roi_matches.csv")
    cached = pd.read_csv(out_csv) if out_csv.exists() and not recompute else pd.DataFrame(columns=["session", "run"])
    done = set(zip(cached.session, cached.run))

    rows = []
    for session, run in list_runs(data_dir):
        if (session, run) in done:
            print(f"[cache] {session} / {run}")
            continue
        try:
            rows.append(run_metrics(session, run, roi_matches))
            print(f"[computed] {session} / {run}")
        except Exception as e:
            print(f"[warn] could not process {session} / {run}: {e}")

    df = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
    return df.sort_values(["session", "run"]).reset_index(drop=True)


def _session_panel(ax, df, metric, title, ylabel, color=FUNC_COLOR, ref=None, median_color=INK_PRIMARY):
    """One dot per run, black bar = session median; sessions in chronological order."""
    sessions = sorted(df.session.unique())
    rng = np.random.default_rng(0)
    for i, ses in enumerate(sessions):
        vals = df.loc[df.session == ses, metric].dropna()
        if vals.empty:
            continue
        x = i + rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(x, vals, s=28, color=color, alpha=0.85, edgecolor="none", zorder=2)
        med = np.median(vals)
        ax.plot([i - 0.3, i + 0.3], [med, med], color=median_color, linewidth=1.8, zorder=3)

    if ref is not None:
        ax.axhline(ref, color=INK_SECONDARY, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels([s.replace("ses-", "") for s in sessions], fontsize=8, rotation=90)
    ax.set_xlim(-0.7, len(sessions) - 0.3)
    ax.set_title(title, fontsize=10.5, color=INK_PRIMARY, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)


def make_figure(df, params=PARAMS):
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.patch.set_facecolor(SURFACE)

    _session_panel(axes[0, 0], df, "n_rois", "ROIs per run", "n ROIs")
    _session_panel(axes[0, 1], df, "frac_rois_matched", "ROIs matched across sessions",
                   "fraction with cluster_id ≥ 0")
    _session_panel(axes[0, 2], df, "frac_nan_rois", "ROIs with NaN dF/F (excluded)", "fraction of ROIs")

    _session_panel(axes[1, 0], df, "median_snr", "Median per-cell SNR", "(p99 − median) / noise SD")
    _session_panel(axes[1, 1], df, "frac_snr_gt_thresh", "Cells with clear transients",
                   f"fraction with SNR > {params['snr_threshold']}")
    _session_panel(axes[1, 2], df, "frac_skew_gt_thresh", "Cells with skewed (transient-like) dF/F",
                   f"fraction with skew > {params['skew_threshold']}")

    _session_panel(axes[2, 0], df, "median_lag1_ac", "Temporal structure",
                   "median lag-1 autocorrelation (0 = white noise)", ref=0)
    _session_panel(axes[2, 1], df, "median_event_frac", "Detected calcium events",
                   f"median fraction of frames > {params['event_sigma']} SD")
    _session_panel(axes[2, 2], df, "median_func_chan2_corr", "Functional vs. channel 2 correlation",
                   "median per-ROI r (high = shared artefact)", ref=0)

    fig.suptitle("Signal quality across sessions (dot = run, bar = session median)",
                 fontsize=11, color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def make_channel2_figure(df, params=PARAMS):
    """Functional (blue) vs. channel 2 (orange) on the same metrics: channel 2 is
    activity-independent, so it should stay low wherever the functional channel is good.
    """
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor(SURFACE)

    for metric, title, ylabel, ax in [
        ("median_snr", "Median per-cell SNR", "(p99 − median) / noise SD", axes[0]),
        ("median_skew", "Median per-cell skew", "skew", axes[1]),
        ("median_lag1_ac", "Temporal structure", "median lag-1 autocorrelation", axes[2]),
    ]:
        _session_panel(ax, df, metric, title, ylabel, color=FUNC_COLOR, median_color=FUNC_COLOR)
        _session_panel(ax, df, "chan2_" + metric, title, ylabel, color=CHAN2_COLOR, median_color=CHAN2_COLOR)
    # a few channel-2 runs have near-zero frame-to-frame noise -> SNR in the hundreds
    axes[0].set_yscale("log")

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=l, markersize=7)
               for c, l in [(FUNC_COLOR, "functional"), (CHAN2_COLOR, "channel 2 (activity-independent)")]]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="ignore cached per-run results")
    parser.add_argument("--out-dir", default=Path(__file__).parent / "qc", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "signal_quality_runs.csv"

    df = gather_results(csv_path, recompute=args.recompute)
    df.to_csv(csv_path, index=False)
    print(f"Saved per-run metrics table to {csv_path}")

    fig = make_figure(df)
    fig.savefig(args.out_dir / "signal_quality.png", dpi=200, facecolor=fig.get_facecolor())
    fig2 = make_channel2_figure(df)
    fig2.savefig(args.out_dir / "signal_quality_chan2.png", dpi=200, facecolor=fig2.get_facecolor())
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
