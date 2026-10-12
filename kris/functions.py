
import numpy as np
import pickle
base = "/ceph/behrens/kris/research/meg_mini2p/"
from scipy.ndimage import gaussian_filter1d

#%% load some data

def get_data(ses, run):
    data = pickle.load(open(f"{base}/data/ses-{ses}/run-{run}.pkl", "rb"))

    rel_cols = np.array([c for c in data.columns if c[:14] == "dff_functional"])
    traces = data[rel_cols].to_numpy()
    traces = (traces - np.mean(traces, axis=0, keepdims = True)) / (np.std(traces, axis=0, keepdims = True) + 1e-6)

    if "event:init_cue_timer" not in data.columns:
        return [[] for _ in range(4)]

    inits = data["event:init_cue_timer"].to_numpy()
    init_times = np.where(inits == 1)[0]
    
    pulses = data["event:pulse_on_timer"].to_numpy()
    pulse_times = np.where(pulses == 1)[0]
    # count multiple pulses as one choice
    pulse_times = np.array([pulse_times[0]] + [pulse_times[i] for i in range(len(pulse_times)) if pulse_times[i] - pulse_times[i-1] > 10])

    # find 'no pulse' choices
    if "event:no_rew_cue_timer" in data.columns:
        no_rew_cues = data["event:no_rew_cue_timer"].to_numpy()
    else: 
        no_rew_cues = np.zeros_like(inits)
    no_rew_times = np.where(no_rew_cues == 1)[0]
    
    # combine into choice times
    choice_times = np.sort(np.concatenate([no_rew_times, pulse_times]))
    
    assert len(init_times) == len(choice_times)
    assert np.all(choice_times > init_times)

    return data, traces, init_times, choice_times

def get_roi_ids():
    
    rois = np.loadtxt(f"{base}/data/roi_matches.csv", dtype=str, delimiter=",", skiprows = 1)

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




#%% construct a data frame from 20 frames before initiation to 20 frames after the choice

def warp_activity(traces, init_times, choice_times, num_bins = 20, num_edge_frames = 4):

    aligned = np.zeros((num_bins+2*num_edge_frames, traces.shape[1], len(choice_times)))

    for i in range(len(choice_times)):
        t0, t1 = init_times[i], choice_times[i]
        #print(i, t0, t1)
        
        aligned[:num_edge_frames, :, i] = traces[t0-num_edge_frames:t0, :]
        aligned[num_edge_frames+num_bins:, :, i] = traces[t1:t1+num_edge_frames, :]
        
        
        bin_edges = np.round(np.linspace(t0-0.1, t1+0.1, num_bins + 1)).astype(int)
        #print(bin_edges)
        
        for bin in range(num_bins):
            ind0 = bin_edges[bin]
            ind1 = bin_edges[bin+1]
            ind1 = ind1+1 if ind1 == ind0 else ind1
            #print(ind0, ind1)
            aligned[num_edge_frames+bin, :, i] = np.mean(traces[ind0:ind1, :], axis=0)
        
    aligned = (aligned - np.mean(aligned, axis=(0,2), keepdims=True)) / (np.std(aligned, axis=(0, 2), keepdims=True) + 1e-6)
    
    return aligned
    
    
    
#%% construct a data frame from just before to just after choice

def aligned_activity(traces, init_times, choice_times, align_to = "choice", bins_before = 15, bins_after = 15, smooth = 3):

    assert len(init_times) == len(choice_times)
    assert np.all(choice_times > init_times)
    
    event_times = choice_times if align_to == "choice" else init_times

    aligned = np.zeros((bins_before + bins_after, traces.shape[1], len(event_times)))

    smooth_traces = traces if smooth == 0 else gaussian_filter1d(traces, smooth, axis=0)

    for i, t in enumerate(event_times):
        #print(i, t)
        aligned[..., i] = smooth_traces[t-bins_before:t+bins_after, :]
        
    return aligned
    