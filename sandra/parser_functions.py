"""Helpers to pull per-run information out of the MEG hackathon pickles.

Each pickle (``ses-XXX/run-YY.pkl``) is a DataFrame with one row per imaging frame,
so all "indices" returned here are frame indices (row positions) into that DataFrame.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

DATA_DIR = Path(
    "/Users/sandrareinert/Library/Mobile Documents/com~apple~CloudDocs/Documents/"
    "Postdoc_SWC/Local_bits/meg_hackathon/data"
)

TRIAL_COLS = [
    "trial_index", "blocks", "trials_in_block", "choice", "chosen_rank",
    "chosen_magnitude", "trial_start_s", "trial_end_s",
]


def load_run(session, run, data_dir=DATA_DIR):
    """Load one run, e.g. ``load_run("ses-230", "run-04")``."""
    return pd.read_pickle(Path(data_dir) / session / f"{run}.pkl")


def list_runs(data_dir=DATA_DIR):
    """All (session, run) pairs available in ``data_dir``."""
    return [(f.parent.name, f.stem) for f in sorted(Path(data_dir).glob("ses-*/run-*.pkl"))]


def event_frames(df, event):
    """Frame indices where ``event:<event>`` fired (empty if the column is missing)."""
    col = f"event:{event}"
    if col not in df.columns:
        return np.array([], dtype=int)
    return np.flatnonzero(df[col].to_numpy() == 1)


def first_reward_frames(df, min_gap=5):
    """First pulse of each reward: pulses closer than ``min_gap`` frames are merged."""
    pulses = event_frames(df, "pulse_on_timer")
    if len(pulses) == 0:
        return pulses
    return pulses[np.insert(np.diff(pulses) > min_gap, 0, True)]


def get_trials(df):
    """One row per logged trial, with init cue and first reward frame.

    ``init_cue_frame`` is the init cue frame labelled with that ``trial_index``.
    ``reward_frame`` is the first reward pulse after the init cue and before the
    next init cue (NaN for unrewarded trials). Some runs have init cues before
    the first logged trial; they are not trials here but still bound the
    reward search.
    """
    cols = [c for c in TRIAL_COLS if c in df.columns]
    trials = df.dropna(subset=["trial_index"]).drop_duplicates("trial_index")[cols].reset_index(drop=True)

    init_cues = event_frames(df, "init_cue_timer")
    cue_trial = df["trial_index"].to_numpy()[init_cues]
    init_frame = []
    for ti in trials["trial_index"]:
        match = init_cues[cue_trial == ti]
        init_frame.append(match[0] if len(match) else np.nan)
    trials["init_cue_frame"] = pd.array(init_frame, dtype="Int64")

    rewards = first_reward_frames(df)
    reward_frame = []
    for start in trials["init_cue_frame"]:
        if pd.isna(start):
            reward_frame.append(np.nan)
            continue
        later_cues = init_cues[init_cues > start]
        stop = later_cues[0] if len(later_cues) else len(df)
        r = rewards[(rewards > start) & (rewards < stop)]
        reward_frame.append(r[0] if len(r) else np.nan)
    trials["reward_frame"] = pd.array(reward_frame, dtype="Int64")
    return trials


def get_F(df, kind="functional"):
    """dF/F traces as (frames x ROIs), plus imaging time (s).

    ``kind`` is ``"functional"`` or ``"channel2"``.
    """
    prefix = f"dff_{kind}_roi_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    return df[cols].to_numpy(), df["imaging_time_s"].to_numpy()


def frame_rate(df):
    return 1 / np.median(np.diff(df["imaging_time_s"].to_numpy()))


def psth(F, event_frames, window=(-2, 2), fs=None, df=None, smooth_sigma=None):
    """Event-aligned traces for every cell.

    Parameters
    ----------
    F : (frames x cells) array
    event_frames : frame indices of the events (NaN / <NA> entries are allowed)
    window : (start, end) in seconds relative to the event
    fs : frame rate in Hz; taken from ``df`` if not given

    Returns
    -------
    traces : (events x cells x time) array; events whose window runs off the
        recording, or whose frame is missing, are all-NaN so rows stay aligned
        with ``event_frames`` (e.g. with the trials table)
    mean : (cells x time) nanmean over events
    t : (time,) seconds relative to the event
    """
    if fs is None:
        if df is None:
            raise ValueError("give either fs or df")
        fs = frame_rate(df)
    offsets = np.arange(int(round(window[0] * fs)), int(round(window[1] * fs)))
    t = offsets / fs

    events = pd.array(event_frames, dtype="Float64").to_numpy(dtype=float, na_value=np.nan)
    traces = np.full((len(events), F.shape[1], len(offsets)), np.nan)
    for i, ev in enumerate(events):
        if np.isnan(ev):
            continue
        idx = int(ev) + offsets
        if idx[0] >= 0 and idx[-1] < len(F):
            traces[i] = gaussian_filter1d(F[idx].T, smooth_sigma, axis=1) if smooth_sigma is not None else F[idx].T

    mean = np.nanmean(traces, axis=0) if np.isfinite(traces).any() else np.full(traces.shape[1:], np.nan)
    return traces, mean, t

def get_roi_ids(data_dir=DATA_DIR):
    rois = np.loadtxt(f"{data_dir}/roi_matches.csv", dtype=str, delimiter=",", skiprows = 1)

    rois_by_session = {}

    ses2uid = {}
    uid2ses = {}

    for line in rois:
        session = str(line[1]).split("_")[0]+"_"+str(line[2]).split("-")[-1]
        if session not in rois_by_session:
            rois_by_session[session] = set()
            ses2uid[session] = {}
            uid2ses[session] = {}
        
        uid = int(line[-1])
        roi_idx = int(line[-4])
        if uid >=0:
            rois_by_session[session].add(int(line[-1]))
            ses2uid[session][roi_idx] = uid
            uid2ses[session][uid] = roi_idx
    return rois_by_session, ses2uid, uid2ses
