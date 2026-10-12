

#%%
import os
os.chdir("/ceph/behrens/kris/research/meg_mini2p/bandit-hackathon/kris")
import pickle
from functions import base, get_data, warp_activity, get_roi_ids, aligned_activity
import numpy as np
import matplotlib.pyplot as plt

rois_by_session, ses2uid, uid2ses = get_roi_ids()

#%%

sess = [219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229]

all_aligned = []
all_runs = {}

for ses in sess:
    runs = np.sort([f[4:6] for f in os.listdir(f"{base}/data/ses-{ses}")])
    all_aligned.append([])
    all_runs[ses] = []
    for run in runs:
        
        data, traces, init_times, choice_times = get_data(ses, run)
        
        if len(init_times) >=5: # at least 5 trials
        
            #aligned = warp_activity(traces, init_times, choice_times)
            aligned = aligned_activity(traces, init_times, choice_times)
            nanfrac = np.mean(np.isnan(aligned.mean((0, 2)))) # fraction of cell with at least one nan
            print("\nsesh =",ses, "run =", run, "aligned shape =", aligned.shape)
            print("nans:", np.mean(np.isnan(aligned)), nanfrac)     
            if nanfrac < 0.05:
                all_aligned[-1].append(aligned)
                all_runs[ses].append(run)

# %% find ROIs shared across all our sessions

run_rois = np.concatenate([[rois_by_session[f"ses-{ses}_{run}"] for run in all_runs[ses]] for ses in sess])
shared_rois = np.sort(list(set.intersection(*run_rois)))

print(len(shared_rois))

#%% extract matched data across sessions

matched = []

for ises, ses in enumerate(sess):
    new_data = []
    for irun, run in enumerate(all_runs[ses]):
        inds = np.array([uid2ses[f"ses-{ses}_{run}"][uid] for uid in shared_rois])
        new_data.append(all_aligned[ises][irun][:, inds])
    matched.append(np.concatenate(new_data, axis = -1))
    print(matched[-1].shape, np.mean(np.isnan(matched[-1])))
    
#%% combine data across matched cells

ind_groups = [[0,1,2,3,4,5], [6,7,8,9,10]]
mean_data = []
for group in ind_groups:
    group_data = np.concatenate([matched[g] for g in group], axis = -1)
    print(group_data.shape, np.mean(np.isnan(group_data)))
    group_data = (group_data - np.nanmean(group_data, axis = (0, 2), keepdims=True)) / (np.nanstd(group_data, axis = (0, 2), keepdims=True) + 1e-6) # zscore
    mean_data.append(np.nanmean(group_data, axis = -1))

# %%

align_by = 1

order = np.argsort(np.argmax(mean_data[align_by], axis = 0))
for igroup, group in enumerate(ind_groups):
    plt.figure()
    plt.imshow(mean_data[igroup][:, order].T, aspect="auto", cmap="viridis")
    plt.axvline(14, color = "r")
    plt.title(f"sessions {group}, aligned by group {ind_groups[align_by]}")
    plt.show()
    plt.close()




# %%
