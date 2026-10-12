
#%%

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d
from functions import base, get_data, warp_activity


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
aligned = warp_activity(traces, init_times, choice_times)
mean_aligned = np.mean(aligned, axis=2)

order = np.argsort(np.argmax(mean_aligned, axis = 0))

plt.figure()
plt.imshow(mean_aligned[:, order].T, aspect="auto", cmap="viridis")
plt.show()
plt.close()

# %%

rois = np.loadtxt(f"{base}/data/roi_matches.csv", dtype=str, delimiter=",", skiprows = 1)


# %%

    
    
#%%


