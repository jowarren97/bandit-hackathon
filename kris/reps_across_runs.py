

#%%
import os
os.chdir("/ceph/behrens/kris/research/meg_mini2p/bandit-hackathon/kris")
import pickle
from functions import base, get_data, warp_activity, get_roi_ids
import numpy as np
import matplotlib.pyplot as plt

rois_by_session, ses2uid, uid2ses = get_roi_ids()

#%%

ses = 223
runs = ["01", "02", "03", "04"]

all_aligned = []

for run in runs:
    
    data, traces, init_times, choice_times = get_data(ses, run)
    
    print(len(init_times), len(choice_times))
    
    aligned = warp_activity(traces, init_times, choice_times)
    mean_aligned = np.mean(aligned, axis=2)
    print(run, mean_aligned.shape)
    
    order = np.argsort(np.argmax(mean_aligned, axis = 0))
    plt.figure()
    plt.imshow(mean_aligned[:, order].T, aspect="auto", cmap="viridis")
    plt.show()
    plt.close()
    
    all_aligned.append(mean_aligned)

# %%

run_rois = [rois_by_session[f"ses-{ses}_{run}"] for run in runs]

shared_rois = np.sort(list(set.intersection(*run_rois)))

# %%

matched = []

for run in runs:
    inds = [uid2ses[f"ses-{ses}_{run}"][uid] for uid in shared_rois]
    print(len(inds))
    matched.append(all_aligned)


# %%
