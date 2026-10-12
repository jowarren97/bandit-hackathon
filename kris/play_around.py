
#%%

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
os.chdir("/ceph/behrens/kris/research/meg_mini2p/bandit-hackathon/kris")
from scipy.ndimage import gaussian_filter1d
from functions import base, get_data, warp_activity, aligned_activity


#%%

data, traces, init_times, choice_times = get_data(223, "01")


#%%


smooth = 10
res = 5
smooth_traces = gaussian_filter1d(traces, smooth, axis=0)[::res, :]

plt.figure()
plt.imshow(smooth_traces.T, aspect="auto", cmap="viridis")
for t in choice_times:
    plt.axvline(t/res, color="r", alpha=0.5)
for t in init_times:
    plt.axvline(t/res, color="y", alpha=0.5)
plt.xlim(400, 1000)
plt.show()
plt.close()

# %%
#aligned = warp_activity(traces, init_times, choice_times)

aligned = aligned_activity(traces, init_times, choice_times)

mean_aligned1 = np.mean(aligned[..., ::2], axis=2)
mean_aligned2 = np.mean(aligned[..., 1::2], axis=2)

order1 = np.argsort(np.argmax(mean_aligned1, axis = 0))
order2 = np.argsort(np.argmax(mean_aligned2, axis = 0))

fig, axs = plt.subplots(1,2, figsize = (10, 5))
axs[0].imshow(mean_aligned1[:, order1].T, aspect="auto", cmap="viridis")
axs[1].imshow(mean_aligned2[:, order1].T, aspect="auto", cmap="viridis")
plt.show()
plt.close()

# %%


# %%

    
    
#%%


