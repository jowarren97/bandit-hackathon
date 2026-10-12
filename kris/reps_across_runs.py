

#%%
import os
os.chdir("/ceph/behrens/kris/research/meg_mini2p/bandit-hackathon/kris")
import pickle
from functions import base, get_data, warp_activity, get_roi_ids, aligned_activity
import numpy as np
import matplotlib.pyplot as plt

rois_by_session, ses2uid, uid2ses = get_roi_ids()

#%%

ses = 224
runs = np.sort([f[4:6] for f in os.listdir(f"{base}/data/ses-{ses}")])

all_aligned = []

for run in runs:
    
    data, traces, init_times, choice_times = get_data(ses, run)
    
    print(len(init_times), len(choice_times), traces.shape)
    
    #aligned = warp_activity(traces, init_times, choice_times)
    aligned = aligned_activity(traces, init_times, choice_times)
    mean_aligned = np.mean(aligned, axis=2)
    print(run, aligned.shape)
    
    order = np.argsort(np.argmax(mean_aligned, axis = 0))
    plt.figure()
    plt.imshow(mean_aligned[:, order].T, aspect="auto", cmap="viridis")
    plt.show()
    plt.close()
    
    all_aligned.append(mean_aligned)

# %%

run_rois = [rois_by_session[f"ses-{ses}_{run}"] for run in runs]
shared_rois = np.sort(list(set.intersection(*run_rois)))

matched = []

for irun, run in enumerate(runs):
    inds = np.array([uid2ses[f"ses-{ses}_{run}"][uid] for uid in shared_rois])
    print(len(inds))
    matched.append(all_aligned[irun][:, inds])

matched = np.array(matched)

# %%

align_by = 1

order = np.argsort(np.argmax(matched[align_by], axis = 0))
for irun, run in enumerate(runs):
    plt.figure()
    plt.imshow(matched[irun][:, order].T, aspect="auto", cmap="viridis")
    plt.title(f"run {run}, aligned by run {runs[align_by]}")
    plt.show()
    plt.close()




# %%
