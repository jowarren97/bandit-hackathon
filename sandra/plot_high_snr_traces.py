"""
For one run per session, plot the dF/F traces of the highest-SNR cells, stacked.

Adapted from Bonsai_ABCD/preprocessing/quality_control/plot_high_snr_traces.py.
This is the visual check behind the summary metrics in compare_signal_quality.py:
does the "best" cell in a session actually show calcium transients, or just noise
that happens to have a high peak-to-noise ratio?

    python plot_high_snr_traces.py [--n-cells 10] [--out-dir qc]

The run per session is the one with the fewest NaN ROIs (ties -> most ROIs), read
from <out-dir>/signal_quality_runs.csv, so run compare_signal_quality.py first.
Cells are ranked by the same SNR as there; if none pass the SNR threshold the top
cells are shown anyway, labelled as such. Traces are in units of each cell's own
noise SD so cells are comparable; reward frames are marked along the top.

One PNG per session is saved to <out-dir>/high_snr_traces/.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_signal_quality import (FUNC_COLOR, GRIDLINE, INK_PRIMARY, INK_SECONDARY, PARAMS,
                                    SURFACE, trace_metrics)
from parser_functions import first_reward_frames, get_F, load_run


def pick_runs(runs):
    """One (session, run) per session: fewest NaN ROIs, then most ROIs."""
    runs = runs.assign(n_valid=runs.n_rois * (1 - runs.frac_nan_rois))
    best = runs.sort_values(["frac_nan_rois", "n_rois"], ascending=[True, False]).groupby("session").head(1)
    return best.sort_values("session")[["session", "run", "n_valid"]]


def plot_session(session, run, n_cells, snr_threshold, out_dir):
    df = load_run(session, run)
    F, t = get_F(df)
    valid = np.flatnonzero(~np.isnan(F).any(axis=0))
    if len(valid) == 0:
        print(f"[warn] skipping {session} / {run}: all ROIs are NaN")
        return

    m = trace_metrics(F[:, valid], PARAMS["event_sigma"])
    order = np.argsort(m["snr"])[::-1][:n_cells]
    n_passing = int(np.sum(m["snr"] > snr_threshold))
    # imaging_time_s has NaNs in a few runs, so build time from frame index
    t = np.arange(len(t)) * np.nanmedian(np.diff(t))

    n_show = len(order)
    offset_step = 12.0  # noise-SD units between stacked traces
    fig, ax = plt.subplots(figsize=(14, max(4, 1.1 * n_show)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # best cell on top
    for i, k in enumerate(order):
        y = (n_show - 1 - i) * offset_step
        trace = F[:, valid[k]]
        trace = (trace - np.median(trace)) / m["noise_sd"][k]
        ax.plot(t, trace + y, color=FUNC_COLOR, linewidth=0.5)
        ax.text(-0.01 * t[-1], y, f"ROI {valid[k]}  SNR={m['snr'][k]:.1f}  skew={m['skew'][k]:.2f}",
                fontsize=8, color=INK_SECONDARY, ha="right", va="center")

    top = n_show * offset_step
    rewards = first_reward_frames(df)
    if len(rewards):
        ax.vlines(t[rewards], top - 0.6 * offset_step, top - 0.2 * offset_step,
                  color=INK_PRIMARY, linewidth=1)
        ax.text(t[-1], top - 0.4 * offset_step, " rewards", fontsize=8, color=INK_SECONDARY, va="center")

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-offset_step, top)
    ax.set_yticks([])
    ax.set_xlabel("time in run (s)  |  amplitude in units of each cell's own noise SD (median-subtracted)",
                  fontsize=9.5, color=INK_SECONDARY)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)

    fallback_note = f"  (no cells pass SNR>{snr_threshold}, showing best available)" if n_passing == 0 else ""
    nan_note = f", {F.shape[1] - len(valid)} NaN ROIs excluded" if len(valid) < F.shape[1] else ""
    ax.set_title(f"{session}  {run}\n"
                 f"top {n_show} cells by SNR ({n_passing}/{len(valid)} pass SNR>{snr_threshold}{nan_note})"
                 f"{fallback_note}",
                 fontsize=10, color=INK_PRIMARY, loc="left")

    fig.tight_layout()
    out_path = out_dir / f"{session}_{run}.png"
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cells", type=int, default=10)
    parser.add_argument("--out-dir", default=Path(__file__).parent / "qc", type=Path)
    args = parser.parse_args()

    runs = pd.read_csv(args.out_dir / "signal_quality_runs.csv")
    out_dir = args.out_dir / "high_snr_traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in pick_runs(runs).iterrows():
        plot_session(row.session, row.run, args.n_cells, PARAMS["snr_threshold"], out_dir)


if __name__ == "__main__":
    main()
