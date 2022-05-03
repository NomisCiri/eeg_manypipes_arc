"""Test the hypotheses specified in the instructions.

Hypotheses read:
There are effects of subsequent memory (i.e., a difference between images that will
be successfully remembered vs. forgotten on a subsequent repetition) ...
a. ... on EEG voltage at any channels, at any time.
b. ... on spectral power, at any frequencies, at any channels, at any time.
"""


import itertools

# %%
# Imports
import os
import pickle
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test, ttest_1samp_no_p
from mne.time_frequency import tfr_morlet
from scipy import stats

from config import (
    FNAME_HYPOTHESES_4_TEMPLATE,
    FPATH_DS,
    OVERWRITE_MSG,
    SUBJS,
    TRIGGER_CODES,
)
from utils import catch, parse_overwrite

# %%
# Path and settings
fpath_ds = FPATH_DS
overwrite = True
fname_report = Path(FNAME_HYPOTHESES_4_TEMPLATE.format(h="h4_report.html"))
fname_h4a = Path(FNAME_HYPOTHESES_4_TEMPLATE.format(h="h4a_cluster.pkl"))
fname_h4b_wavelet = Path(FNAME_HYPOTHESES_4_TEMPLATE.format(h="h4b_wavelet.pkl"))
fname_h4b_cluster = Path(FNAME_HYPOTHESES_4_TEMPLATE.format(h="h4b_cluster.pkl"))

# Settings for cluster test
tfce = dict(start=0, step=0.2)
p_accept = 0.05
sigma = 1e-3  # sigma for the "hat" method
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
threshold = stats.distributions.t.ppf(1 - p_accept, len(SUBJS) - 1)  # threshold

# Time frequency
freqs = np.logspace(*np.log10([4, 100]), num=40).round()
n_cycles = freqs / 2.0  # different number of cycle per frequency
n_cycles.round()
# toi
toi_min = -0.2
toi_max = 1.5
# List of all trigger combinations for a new image
triggers_hits_list = list(
    itertools.product(
        list(TRIGGER_CODES[0].values()),
        list(TRIGGER_CODES[1].values()),
        list(TRIGGER_CODES[2].values()),
        ["sub_remembered"],
    )
)
# List of all trigger combinations for an old image
triggers_misses_list = list(
    itertools.product(
        list(TRIGGER_CODES[0].values()),
        list(TRIGGER_CODES[1].values()),
        list(TRIGGER_CODES[2].values()),
        ["sub_forgotten"],
    )
)
# %%
# Makes triggercodes for subsetting the epochs
triggers_hits = [
    "/".join(map(str, triggers_hits_list[i])) for i in range(0, len(triggers_hits_list))
]
triggers_misses = [
    "/".join(map(str, triggers_misses_list[i]))
    for i in range(0, len(triggers_misses_list))
]
# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        fpath_ds=fpath_ds,
        overwrite=overwrite,
    )
    defaults = parse_overwrite(defaults)
    fpath_ds = defaults["fpath_ds"]
    overwrite = defaults["overwrite"]
# %%
# Check overwrite
if fname_report.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_report))
# %%
# Start a report to save figures
report = mne.Report(title="Hypotheses 3")
# %%
# Reads in all epochs
epochs = [
    catch(
        lambda: mne.read_epochs(
            fname=os.path.join(
                str(fpath_ds),
                "derivatives",
                f"EMP{sub:02}",
                f"EMP{sub:02}_clean-epo.fif.gz",
            )
        )
    )
    for sub in SUBJS
]
#  Keep only existing subs
epochs_complete = list(filter(None.__ne__, epochs))
# %%
# Get a list of epochs in the desired timerange and with the desired channels.
# already put it into the format needed for permutation test
# required format: (n_observations (subs), time, n_vertices (channels)).
evokeds_diff_list = list(
    [
        np.subtract(
            x[triggers_hits].crop(toi_min, toi_max).average().get_data(),
            x[triggers_misses].crop(toi_min, toi_max).average().get_data(),
        )
        for x in epochs_complete
    ]
)
# add list elements along array axis and reshape for permutation test
evokeds_diff_arr = np.stack(evokeds_diff_list, axis=2).transpose(2, 1, 0)
# Concatanate conditions for use with cluster based permutation test
# %%
# Calculate adjacency matrix between sensors from their locations
sensor_adjacency, ch_names_theta = find_ch_adjacency(
    epochs_complete[1].copy().info, "eeg"
)
# %%
# Calculate statistical thresholds, h4a confirmed
# Check overwrite
# If there is a cluster test, and overwrite is false, load data
if fname_h4a.exists() and not overwrite:
    file = open(fname_h4a, "rb")
    clusterstats = pickle.load(file)
    file.close()
# If overwriting is false compute everything again
else:
    clusterstats = spatio_temporal_cluster_1samp_test(
        evokeds_diff_arr,
        threshold=threshold,
        n_permutations=1000,
        adjacency=sensor_adjacency,
        n_jobs=40,
    )
    file = open(fname_h4a, "wb")
    pickle.dump(clusterstats, file)
    file.close()

t_obs_h4a, clusters_h4a, cluster_pv_h4a, h0_h4a = clusterstats

significant_points_h4a = cluster_pv_h4a.reshape(t_obs_h4a.shape).T < 0.05

# %%
# Visualize the voltage, taking the average of all subjects
# make new object so mne plotting routines can be used
epochs_misses_plot = list([epo[triggers_misses].average() for epo in epochs_complete])
epochs_hits_plot = list([epo[triggers_hits].average() for epo in epochs_complete])
# calculate difference wave
evoked = mne.combine_evoked(
    [mne.grand_average(epochs_misses_plot), mne.grand_average(epochs_hits_plot)],
    weights=[1, -1],
).crop(toi_min, toi_max)
time_unit = dict(time_unit="s")
# show difference wave
joint = evoked.plot_joint(
    title="Misses - Hits",
    ts_args=time_unit,
    # times=[0.3, 0.35, 0.4, 0.45, 0.5],
    topomap_args=time_unit,
)
report.add_figure(
    fig=joint,
    title="h4a",
    caption="This figure shows a difference in voltage at image presentation."
    + "The difference is computed between old images that"
    + "have been correctly recognized as old and those that have been falsely"
    + "recognized as new. All eeg channels are shown",
    image_format="PNG",
)
# Create ROIs by checking channel labels
# only check tois
# Visualize the results
toi_evoked = evoked.copy().crop(toi_min, toi_max)
h4a_test = toi_evoked.plot_image(
    colorbar=False,
    show=False,
    mask=significant_points_h4a,
    show_names="all",
    titles="Significant timepoints",
    **time_unit,
)
h4a_test.set_figheight(15)
report.add_figure(
    fig=h4a_test,
    title="h4a sig",
    caption="This figure shows where the difference between old and new"
    + "image presentation are significant according"
    + "to a cluster based permutation test."
    + "Only greyscales imply that there is"
    + "no significant difference in the time window of interest",
    image_format="PNG",
)
# %%
# Hypothesis 3b.
# Do wavelet tranformation on whole epoch to get tfr
# If there is a wavelet file test, and overwrite is false, load data
if fname_h4b_wavelet.exists() and not overwrite:
    file_wavelet = open(fname_h4b_wavelet, "rb")
    tfr_diff_list = pickle.load(file)
    file.close()
else:
    tfr_diff_list = list(
        [
            np.subtract(
                tfr_morlet(
                    x[triggers_hits],
                    freqs,
                    n_cycles=n_cycles,
                    average=True,
                    return_itc=False,
                    n_jobs=6,
                )
                .crop(toi_min, toi_max)
                .data,
                tfr_morlet(
                    x[triggers_misses],
                    freqs,
                    n_cycles=n_cycles,
                    average=True,
                    return_itc=False,
                    n_jobs=6,
                )
                .crop(toi_min, toi_max)
                .data,
            )
            for x in epochs_complete
        ]
    )
    file = open(fname_h4b_wavelet, "wb")
    pickle.dump(tfr_diff_list, file)
    file.close()
# %%
# Concatanate conditions for use with cluster based permutation test
# required format: (n_observations (subs),freq, time, n_vertices (channels)).
tfr_diff_arr = np.stack(tfr_diff_list, axis=2).transpose(2, 1, 3, 0)
# %%
# Make sensor-frequency adjacancy matrix
tf_timepoints = tfr_diff_arr.shape[2]
tfr_adjacency = mne.stats.combine_adjacency(len(freqs), tf_timepoints, sensor_adjacency)
# %%
# do clusterstats
# If there is a cluster test filse, and overwrite is false, load data
if fname_h4b_cluster.exists() and not overwrite:
    file_cluster = open(fname_h4b_cluster, "rb")
    clusterstats_h4b = pickle.load(file)
    file.close()
else:
    clusterstats = spatio_temporal_cluster_1samp_test(
        tfr_diff_arr,
        threshold=threshold,
        n_permutations=1000,
        adjacency=tfr_adjacency,
        n_jobs=40,
        stat_fun=stat_fun_hat,
    )
    t_obs_diff_h4b, clusters_diff_h4b, cluster_pv_diff_h4b, h0_diff_h4b = clusterstats
    file_h4b_cluster = open(fname_h4b_cluster, "wb")
    pickle.dump(clusterstats, file_h4b_cluster)
    file_h4b_cluster.close()
significant_points_diff_h4b = np.where(cluster_pv_diff_h4b < 0.05)[0]
# %%
# calculate average power difference
tfr_theta_diff = np.average(tfr_diff_arr, axis=0).transpose(1, 0, 2)
t_obs_diff_h4b_t = t_obs_diff_h4b.transpose(1, 0, 2)
# %%
# make h4b figure for every channel.
h4b_test, axs = plt.subplots(
    nrows=len(ch_names_theta), ncols=2, figsize=(200, 20), constrained_layout=True
)

for ch_idx in range(0, len(ch_names_theta)):
    plt.sca(axs[ch_idx, 0])
    plt.imshow(
        tfr_theta_diff[:, :, ch_idx],
        aspect="auto",
        origin="lower",
        extent=[toi_min, toi_max, freqs[0], freqs[-1]],
    )
    plt.colorbar()
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Power difference new - old \n ({ch_names_theta[ch_idx]})")

    plt.sca(axs[ch_idx, 1])
    plt.imshow(
        t_obs_diff_h4b_t[:, :, ch_idx],
        aspect="auto",
        origin="lower",
        extent=[toi_min, toi_max, freqs[0], freqs[-1]],
    )
    plt.colorbar()
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Cluster T_val difference new -old \n ({ch_names_theta[ch_idx]})")

# add to report
report.add_figure(
    fig=h4b_test,
    title="h4a sig",
    caption="This figure shows where the difference in theta"
    + "power between old and new images. The first column shows "
    + "The first column shows raw power  difference, the second "
    + "show the corresponding T-statistic",
    image_format="PNG",
)

# %%
# make dummy tfr for figure
tfr_specs = tfr_morlet(
    epochs_complete[0][triggers_hits],
    freqs,
    n_cycles=n_cycles,
    average=True,
    return_itc=False,
    n_jobs=6,
).crop(toi_min, toi_max)

# %% save report
report.save(fname_report, overwrite=overwrite)

# %%
