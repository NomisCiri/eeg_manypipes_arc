"""Test the hypotheses specified in the instructions.

> There are effects of image novelty
> (i.e., between images shown for the first time/new vs. repeated/old images)
> within the time-range from 300-500 ms ...
> a. ... on EEG voltage at fronto-central channels.
> b. ... on theta power at fronto-central channels.
> c. ... on alpha power at posterior channels.

"""

# %%
# Imports
import itertools
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
    FNAME_EPO_CLEAN_TEMPLATE,
    FNAME_HYPOTHESES_2_TEMPLATE,
    FNAME_REPORT_H2,
    OVERWRITE_MSG,
    SUBJS,
    TRIGGER_CODES,
)
from utils import parse_overwrite

# %%
# Filepaths and settings
overwrite = True
fname_report = FNAME_REPORT_H2
fname_h2a = Path(FNAME_HYPOTHESES_2_TEMPLATE.format(h="h2a_cluster.pkl"))
fname_h2b_wavelet = Path(FNAME_HYPOTHESES_2_TEMPLATE.format(h="h2b_wavelet.pkl"))
fname_h2b_cluster = Path(FNAME_HYPOTHESES_2_TEMPLATE.format(h="h2b_cluster.pkl"))
fname_h2c_wavelet = Path(FNAME_HYPOTHESES_2_TEMPLATE.format(h="h2c_wavelet.pkl"))
fname_h2c_cluster = Path(FNAME_HYPOTHESES_2_TEMPLATE.format(h="h2c_cluster.pkl"))

# Make sure the save directories exist
for _fname in [
    fname_h2a,
    fname_h2b_wavelet,
    fname_h2b_cluster,
    fname_h2c_wavelet,
    fname_h2c_cluster,
]:
    _fname.parent.mkdir(parents=True, exist_ok=True)

# Settings for cluster test
pthresh = 0.05  # general significance alpha level
pthresh_cluster = 0.001  # cluster forming alpha level
tail = 0  # two-tailed, see also "pthresh / 2" below
sigma = 1e-3  # sigma for the small variance correction
thresh = stats.distributions.t.ppf(1 - pthresh_cluster / 2, len(SUBJS) - 1)
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
nperm = 5000
seed_H2 = 1991

# Time frequency
n_cycles = 7
alpha_freqs = np.arange(8, 12.5, 0.5)  # define frequencies of interest
theta_freqs = np.arange(4, 7.5, 0.5)  # define frequencies of interest
# Roi & toi
ch_fronto_central = ["FC3", "FC4", "Fz", "FC1", "FC2"]
ch_posterior = ["POz", "PO3", "PO4", "Oz", "O1", "O2", "PO7", "PO8"]
toi_min = 0.3
toi_max = 0.5
# List of all trigger combinations for a new image
triggers_new_list = list(
    itertools.product(
        list(TRIGGER_CODES[0].values()),
        ["new"],
        list(TRIGGER_CODES[2].values()),
        list(TRIGGER_CODES[3].values()),
    )
)
# List of all trigger combinations for an old image
triggers_old_list = list(
    itertools.product(
        list(TRIGGER_CODES[0].values()),
        ["old"],
        list(TRIGGER_CODES[2].values()),
        list(TRIGGER_CODES[3].values()),
    )
)
# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        overwrite=overwrite,
    )
    defaults = parse_overwrite(defaults)
    overwrite = defaults["overwrite"]

# %%
# Check overwrite
if fname_report.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_report))

# %%
# Start a report to save figures
report = mne.Report(title="Hypothesis 2")

# %%
# Makes triggercodes for subsetting the epochs
triggers_new = [
    "/".join(map(str, triggers_new_list[i])) for i in range(0, len(triggers_new_list))
]
triggers_old = [
    "/".join(map(str, triggers_old_list[i])) for i in range(0, len(triggers_old_list))
]
# %%
# Reads in all epochs
epochs_complete = [
    mne.read_epochs(fname=FNAME_EPO_CLEAN_TEMPLATE.format(sub=sub)) for sub in SUBJS
]

# %%
# Get a list of epochs in the desired time range and with the desired channels.
# already put it into the format needed for permutation test
# required format: (n_observations (subs), time, n_vertices (channels)).
evokeds_diff_list = list(
    [
        np.subtract(
            x[triggers_old]
            .filter(h_freq=40, l_freq=None)
            .apply_baseline(baseline=(None, 0))
            .crop(toi_min, toi_max)
            .pick_channels(ch_fronto_central)
            .average()
            .get_data(),
            x[triggers_new]
            .filter(h_freq=40, l_freq=None)
            .apply_baseline(baseline=(None, 0))
            .crop(toi_min, toi_max)
            .pick_channels(ch_fronto_central)
            .average()
            .get_data(),
        )
        for x in epochs_complete
    ]
)
# add list elements along array axis and reshape for permutation test
# Concatanate conditions for use with cluster based permutation test
evokeds_diff_arr = np.stack(evokeds_diff_list, axis=2).transpose(2, 1, 0)

# %%
# Calculate adjacency matrix between sensors from their locations
sensor_adjacency, ch_names = find_ch_adjacency(
    epochs_complete[1].copy().pick_channels(ch_fronto_central).info, "eeg"
)
# %%
# Calculate statistical thresholds
# Check overwrite
# If there is a cluster test, and overwrite is false, load data
if fname_h2a.exists() and not overwrite:
    with open(fname_h2a, "rb") as fin:
        clusterstats = pickle.load(fin)

# If overwriting is false compute everything again
else:
    clusterstats = spatio_temporal_cluster_1samp_test(
        evokeds_diff_arr,
        threshold=thresh,
        n_permutations=nperm,
        adjacency=sensor_adjacency,
        n_jobs=6,
        stat_fun=stat_fun_hat,
        out_type="mask",
        tail=tail,
        seed=seed_H2,
    )
    with open(fname_h2a, "wb") as fout:
        pickle.dump(clusterstats, fout)

t_obs_h2a, clusters_h2a, cluster_pv_h2a, h0_h2a = clusterstats

# %%
# Visualize the voltage, taking the average of all subjects
# old images
epochs_old_plot = list(
    [
        epo[triggers_old].pick_channels(ch_fronto_central).average()
        for epo in epochs_complete
    ]
)
epochs_new_plot = list(
    [
        epo[triggers_new].pick_channels(ch_fronto_central).average()
        for epo in epochs_complete
    ]
)
# calculate difference wave
evoked = mne.combine_evoked(
    [mne.grand_average(epochs_old_plot), mne.grand_average(epochs_new_plot)],
    weights=[1, -1],
)
time_unit = dict(time_unit="s")
# show difference wave
joint = evoked.plot_joint(
    title="Old - New images",
    ts_args=time_unit,
    times=[0.3, 0.35, 0.4, 0.45, 0.5],
    topomap_args=time_unit,
)
report.add_figure(
    fig=joint,
    title="h2a",
    caption="This figure shows the difference between old and new "
    + "image presentation as well as the scalp distribution in the times of interest",
    image_format="PNG",
)

# Create ROIs by checking channel labels
# only check tois
# Visualize the results
toi_evoked = evoked.copy().crop(toi_min, toi_max)
h2a_test = toi_evoked.plot_image(
    colorbar=False,
    show=False,
    # mask=clusters_h2a[0],
    show_names="all",
    titles="Significant timepoints",
    **time_unit,
)

report.add_figure(
    fig=h2a_test,
    title="h2a sig",
    caption="This figure shows where the difference between old and new"
    + "image presentation are significant according"
    + "to a cluster based permutation test."
    + "Only greyscales imply that there is"
    + "no significant difference in the time window of interest",
    image_format="PNG",
)

# %%
# get times and sensors of signifcant clusters

# get cluster defining start time and end time
cluster_time_h2a = [
    [
        np.min(
            np.asarray(toi_evoked.times)[np.unique(np.where(clusters_h2a[clusters])[0])]
        ),  # get min time of cluster
        np.max(
            np.asarray(toi_evoked.times)[np.unique(np.where(clusters_h2a[clusters])[0])]
        ),  # get max time of cluster
    ]
    for clusters in range(0, len(clusters_h2a))
]


# get cluster defining channels
cluster_chs_h2a = [
    np.asarray(ch_fronto_central)[np.unique(np.where(clusters_h2a[clusters])[1])]
    for clusters in range(0, len(clusters_h2a))
]

# %%
# Hypothesis 2b.
# Do wavelet tranformation on whole epoch to get tfr
# If there is a wavelet file, and overwrite is false, load data
# Note: apply baseline after TF decomposition
# (https://www.youtube.com/watch?v=9dXG50ychsQ)
if fname_h2b_wavelet.exists() and not overwrite:
    with open(fname_h2b_wavelet, "rb") as fin:
        tfr_diff_list = pickle.load(fin)

else:
    tfr_diff_h2b_list = list(
        [
            np.subtract(
                tfr_morlet(
                    x[triggers_new].pick_channels(ch_fronto_central),
                    theta_freqs,
                    n_cycles=n_cycles,
                    average=True,
                    return_itc=False,
                    n_jobs=6,
                )
                .apply_baseline(baseline=(None, -0.1))
                .crop(toi_min, toi_max)
                .data,
                tfr_morlet(
                    x[triggers_old].pick_channels(ch_fronto_central),
                    theta_freqs,
                    n_cycles=n_cycles,
                    average=True,
                    return_itc=False,
                    n_jobs=6,
                )
                .apply_baseline(baseline=(None, -0.1))
                .crop(toi_min, toi_max)
                .data,
            )
            for x in epochs_complete
        ]
    )
    with open(fname_h2b_wavelet, "wb") as fout:
        pickle.dump(tfr_diff_h2b_list, fout)

# %%
# Concatanate conditions for use with cluster based permutation test
# required format: (n_observations (subs),freq, time, n_vertices (channels)).
tfr_theta_diff_arr = np.stack(tfr_diff_h2b_list, axis=2).transpose(2, 1, 3, 0)

# %%
# Make sensor-frequency adjacancy matrix
tf_timepoints = tfr_theta_diff_arr.shape[2]
tfr_adjacency = mne.stats.combine_adjacency(
    len(theta_freqs), tf_timepoints, sensor_adjacency
)

# %%
# Calculate statistical thresholds
if fname_h2b_cluster.exists() and not overwrite:
    with open(fname_h2b_cluster, "rb") as fin:
        clusterstats = pickle.load(fin)

else:
    clusterstats = spatio_temporal_cluster_1samp_test(
        tfr_theta_diff_arr,
        threshold=thresh,
        n_permutations=nperm,
        adjacency=tfr_adjacency,
        n_jobs=6,
        stat_fun=stat_fun_hat,
        out_type="mask",
        tail=tail,
        seed=seed_H2,
    )
    with open(fname_h2b_cluster, "wb") as fout:
        pickle.dump(tfr_diff_h2b_list, fout)

t_obs_h2b, clusters_h2b, cluster_pv_h2b, h0_h2b = clusterstats


# %%
# calculate power difference
tfr_theta_diff = np.average(tfr_theta_diff_arr, axis=0)
t_obs_h2b_t = t_obs_h2b.transpose(1, 0, 2)

# %%
# make h2b figure
h2b_test, axs = plt.subplots(
    nrows=len(ch_names), ncols=2, figsize=(20, 20), constrained_layout=True
)

for ch_idx in range(0, len(ch_names)):
    plt.sca(axs[ch_idx, 0])
    plt.imshow(
        tfr_theta_diff[:, :, ch_idx],
        aspect="auto",
        origin="lower",
        extent=[toi_min, toi_max, theta_freqs[0], theta_freqs[-1]],
    )
    plt.colorbar()
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Power difference new - old \n ({ch_names[ch_idx]})")

    plt.sca(axs[ch_idx, 1])
    plt.imshow(
        t_obs_h2b_t[:, :, ch_idx],
        aspect="auto",
        origin="lower",
        extent=[toi_min, toi_max, theta_freqs[0], theta_freqs[-1]],
    )
    plt.colorbar()
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Cluster T_val difference new -old \n ({ch_names[ch_idx]})")

report.add_figure(
    fig=h2b_test,
    title="h2a sig",
    caption="This figure shows where the difference in theta"
    + "power between old and new images. The first column shows "
    + "The first column shows raw power  difference, the second "
    + "show the corresponding T-statistic",
    image_format="PNG",
)
# %%
# get times, sensors and frequencies of signifcant clusters

# get cluster defining start time and end time
cluster_time_h2b = [
    [
        np.min(
            np.asarray(toi_evoked.times)[
                np.unique(np.where(clusters_h2b[clusters_h2b_idx])[1])
            ]
        ),  # get min time of cluster
        np.max(
            np.asarray(toi_evoked.times)[
                np.unique(np.where(clusters_h2b[clusters_h2b_idx])[1])
            ]
        ),  # get max time of cluster
    ]
    for clusters_h2b_idx in range(0, len(clusters_h2b))
]

# get cluster defining sensors
cluster_chs_h2b = [
    np.asarray(ch_fronto_central)[
        np.unique(np.where(clusters_h2b[clusters_h2b_idx])[2])
    ]
    for clusters_h2b_idx in range(0, len(clusters_h2b))
]
# get cluster defining freqs
cluster_freqs_h2b = [
    np.asarray(theta_freqs)[np.unique(np.where(clusters_h2b[clusters_h2b_idx])[0])]
    for clusters_h2b_idx in range(0, len(clusters_h2b))
]

# %%
# Hypothesis 2c.
# Do wavelet tranformation on whole epoch to get tfr
# If there is a wavelet file, and overwrite is false, load data
if fname_h2c_wavelet.exists() and not overwrite:
    with open(fname_h2c_wavelet, "rb") as fin:
        tfr_diff_h2c_list = pickle.load(fin)

else:
    tfr_diff_h2c_list = list(
        [
            np.subtract(
                tfr_morlet(
                    x[triggers_new].pick_channels(ch_posterior),
                    alpha_freqs,
                    n_cycles=n_cycles,
                    average=True,
                    return_itc=False,
                    n_jobs=6,
                )
                .apply_baseline(baseline=(None, -0.1))
                .crop(toi_min, toi_max)
                .data,
                tfr_morlet(
                    x[triggers_old].pick_channels(ch_posterior),
                    alpha_freqs,
                    n_cycles=n_cycles,
                    average=True,
                    return_itc=False,
                    n_jobs=6,
                )
                .apply_baseline(baseline=(None, -0.1))
                .crop(toi_min, toi_max)
                .data,
            )
            for x in epochs_complete
        ]
    )
    with open(fname_h2c_wavelet, "wb") as fout:
        pickle.dump(tfr_diff_h2c_list, fout)

# %%
# Concatanate conditions for use with cluster based permutation test
# required format: (n_observations (subs),freq, time, n_vertices (channels)).
tfr_alpha_diff_arr = np.stack(tfr_diff_h2c_list, axis=2).transpose(2, 1, 3, 0)

# %%
# Make sensor-frequency adjacancy matrix for alpha channels
sensor_adjacency_alpha, ch_names_alpha = find_ch_adjacency(
    epochs_complete[1].copy().pick_channels(ch_posterior).info, "eeg"
)
tf_timepoints_alpha = tfr_alpha_diff_arr.shape[2]
tfr_adjacency_alpha = mne.stats.combine_adjacency(
    len(alpha_freqs), tf_timepoints_alpha, sensor_adjacency_alpha
)

# %%
# Calculate statistical thresholds
if fname_h2c_cluster.exists() and not overwrite:
    with open(fname_h2c_cluster, "rb") as fin:
        clusterstats_h2c = pickle.load(fin)

else:
    clusterstats_h2c = spatio_temporal_cluster_1samp_test(
        tfr_alpha_diff_arr,
        threshold=thresh,
        n_permutations=nperm,
        adjacency=tfr_adjacency_alpha,
        stat_fun=stat_fun_hat,
        out_type="mask",
        tail=tail,
        seed=seed_H2,
        n_jobs=6,
    )
    with open(fname_h2c_cluster, "wb") as fout:
        pickle.dump(tfr_diff_h2c_list, fout)

t_obs_h2c, clusters_h2c, cluster_pv_h2c, h0_h2c = clusterstats_h2c
significant_points_h2c = np.where(cluster_pv_h2c < pthresh)[0]

# %%
# calculate power difference
tfr_alpha_diff = np.average(tfr_alpha_diff_arr, axis=0)
t_obs_h2c_t = t_obs_h2c.transpose(1, 0, 2)
# %%
# make h2b figure
h2c_test, axs = plt.subplots(
    nrows=len(ch_names_alpha), ncols=2, figsize=(20, 20), constrained_layout=True
)

for ch_idx in range(0, len(ch_names_alpha)):
    # power difference
    plt.sca(axs[ch_idx, 0])
    plt.imshow(
        tfr_alpha_diff[:, :, ch_idx],
        aspect="auto",
        origin="lower",
        extent=[toi_min, toi_max, alpha_freqs[0], alpha_freqs[-1]],
    )
    plt.colorbar()
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Power difference new - old \n ({ch_names_alpha[ch_idx]})")

    # T values
    plt.sca(axs[ch_idx, 1])
    plt.imshow(
        t_obs_h2c_t[:, :, ch_idx],
        aspect="auto",
        origin="lower",
        extent=[toi_min, toi_max, alpha_freqs[0], alpha_freqs[-1]],
    )
    plt.colorbar()
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Cluster T_val difference new -old \n ({ch_names_alpha[ch_idx]})")


report.add_figure(
    fig=h2c_test,
    title="h2a sig",
    caption="This figure shows where the difference in alpha"
    + "power between old and new images. The first column shows "
    + "The first column shows raw power  difference, the second "
    + "show the corresponding T-statistic",
    image_format="PNG",
)

# %%
# get times, sensors and frequencies of signifcant clusters

# get cluster defining start time and end time
cluster_time_h2c = [
    [
        np.min(
            np.asarray(toi_evoked.times)[np.unique(np.where(clusters_h2c[clusters])[1])]
        ),  # get min time of cluster
        np.max(
            np.asarray(toi_evoked.times)[np.unique(np.where(clusters_h2c[clusters])[1])]
        ),  # get max time of cluster
    ]
    for clusters in range(0, len(clusters_h2c))
]
# get cluster defining sensors
cluster_chs_h2c = [
    np.asarray(ch_posterior)[np.unique(np.where(clusters_h2c[clusters_h2c_idx])[2])]
    for clusters_h2c_idx in range(0, len(clusters_h2c))
]
# get cluster defining freqs
cluster_freqs_h2c = [
    np.asarray(alpha_freqs)[np.unique(np.where(clusters_h2b[clusters_h2c_idx])[0])]
    for clusters_h2c_idx in range(0, len(clusters_h2c))
]

# %%
# Save report
report.save(fname_report, open_browser=False, overwrite=overwrite)

# %%
