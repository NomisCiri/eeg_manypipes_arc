"""Test the hypotheses specified in the instructions.

Hypotheses read:
There are effects of image novelty (i.e., between images shown for the first time/new
vs. repeated/old images) within the time-range from 300–500 ms ...
a. ... on EEG voltage at fronto-central channels.
b. ... on theta power at fronto-central channels.
c. ... on alpha power at posterior channels.
"""

import itertools

# %%
# Imports
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_test
from mne.time_frequency import tfr_morlet

from config import (
    FNAME_REPORT_HYPOTHESES_TEMPLATE,
    FPATH_DS,
    OVERWRITE_MSG,
    SUBJS,
    TRIGGER_CODES,
)
from utils import catch

# %%
# Path and settings
fpath_ds = FPATH_DS
overwrite = True
# Settings for cluster test
tfce = dict(start=0, step=0.2)
p_accept = 0.05
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
# Check overwrite
fname_report = Path(FNAME_REPORT_HYPOTHESES_TEMPLATE.format(h="h2"))
if fname_report.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_report))
# %%
# Start a report to save figures
report = mne.Report(title="Hypotheses 2")
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
# %%
#  Keep only existing subs
epochs_complete = list(filter(None.__ne__, epochs))
# %%
# Get a list of epochs in the desired timerange and with the desired channels.
# already put it into the format needed for permutation test
# required format: (n_observations (subs), time, n_vertices (channels)).

# old images
evokeds_old_list = list(
    [
        x[triggers_old]
        .crop(toi_min, toi_max)
        .pick_channels(ch_fronto_central)
        .average()
        .get_data()
        for x in epochs_complete
    ]
)
# new images
evokeds_new_list = list(
    [
        x[triggers_new]
        .crop(toi_min, toi_max)
        .pick_channels(ch_fronto_central)
        .average()
        .get_data()
        for x in epochs_complete
    ]
)
# add list elements along array axis and reshape for permutation test
evokeds_new_arr = np.stack(evokeds_new_list, axis=2).transpose(2, 1, 0)
evokeds_old_arr = np.stack(evokeds_old_list, axis=2).transpose(2, 1, 0)
# Concatanate conditions for use with cluster based permutation test
X_h2a = [evokeds_old_arr, evokeds_new_arr]
# %%
# Calculate adjacency matrix between sensors from their locations
sensor_adjacency, ch_names_theta = find_ch_adjacency(
    epochs_complete[1].copy().pick_channels(ch_fronto_central).info, "eeg"
)
# %%
# Calculate statistical thresholds, h2a not confirmed
t_obs_h2a, clusters_h2a, cluster_pv_h2a, h0_h2a = spatio_temporal_cluster_test(
    X_h2a, tfce, n_permutations=1000, adjacency=sensor_adjacency
)
significant_points_h2a = cluster_pv_h2a.reshape(t_obs_h2a.shape).T < 0.05
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
    mask=significant_points_h2a,
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
# Hypothesis 2b.
# Do wavelet tranformation on whole epoch to get tfr
tfr_theta_new_list = list(
    [
        tfr_morlet(
            x[triggers_new].pick_channels(ch_fronto_central),
            theta_freqs,
            n_cycles=n_cycles,
            average=True,
            return_itc=False,
            n_jobs=1,
        )
        .crop(toi_min, toi_max)
        .data
        for x in epochs_complete
    ]
)

tfr_theta_old_list = list(
    [
        tfr_morlet(
            x[triggers_old].pick_channels(ch_fronto_central),
            theta_freqs,
            n_cycles=n_cycles,
            average=True,
            return_itc=False,
            n_jobs=1,
        )
        .crop(toi_min, toi_max)
        .data
        for x in epochs_complete
    ]
)
# %%
# Concatanate conditions for use with cluster based permutation test
# required format: (n_observations (subs),freq, time, n_vertices (channels)).
tfr_theta_new_arr = np.stack(tfr_theta_new_list, axis=2).transpose(2, 1, 3, 0)
tfr_theta_old_arr = np.stack(tfr_theta_old_list, axis=2).transpose(2, 1, 3, 0)
X_h2b = [tfr_theta_new_arr, tfr_theta_old_arr]
# %%
# Make sensor-frequency adjacancy matrix
tf_timepoints = tfr_theta_new_arr.shape[2]
tfr_adjacency = mne.stats.combine_adjacency(
    len(theta_freqs), tf_timepoints, sensor_adjacency
)
# %%
# Calculate statistical thresholds, h2b not confirmed
t_obs_h2b, clusters_h2b, cluster_pv_h2b, h0_h2b = spatio_temporal_cluster_test(
    X_h2b, tfce, n_permutations=1000, adjacency=tfr_adjacency
)
significant_points_h2b = np.where(cluster_pv_h2b < p_accept)[0]
# %%
# calculate power difference
grand_avg_tfr_old = np.average(tfr_theta_old_arr, axis=0)
grand_avg_tfr_new = np.average(tfr_theta_new_arr, axis=0)
tfr_theta_diff = np.subtract(grand_avg_tfr_new, grand_avg_tfr_old).transpose(1, 0, 2)
t_obs_h2b_t = t_obs_h2b.transpose(1, 0, 2)
# %%
# make h2b figure
h2b_test, axs = plt.subplots(
    nrows=len(ch_names_theta), ncols=2, figsize=(20, 20), constrained_layout=True
)

for ch_idx in range(0, len(ch_names_theta)):
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
    plt.title(f"Power difference new - old \n ({ch_names_theta[ch_idx]})")

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
    plt.title(f"Cluster T_val difference new -old \n ({ch_names_theta[ch_idx]})")


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
# Hypothesis 2c.
# Do wavelet tranformation on whole epoch to get tfr
tfr_alpha_new_list = list(
    [
        tfr_morlet(
            x[triggers_new].pick_channels(ch_posterior),
            alpha_freqs,
            n_cycles=n_cycles,
            average=True,
            return_itc=False,
            n_jobs=1,
        )
        .crop(toi_min, toi_max)
        .data
        for x in epochs_complete
    ]
)

tfr_alpha_old_list = list(
    [
        tfr_morlet(
            x[triggers_old].pick_channels(ch_posterior),
            alpha_freqs,
            n_cycles=n_cycles,
            average=True,
            return_itc=False,
            n_jobs=1,
        )
        .crop(toi_min, toi_max)
        .data
        for x in epochs_complete
    ]
)
# %%
# Concatanate conditions for use with cluster based permutation test
# required format: (n_observations (subs),freq, time, n_vertices (channels)).
tfr_alpha_new_arr = np.stack(tfr_alpha_new_list, axis=2).transpose(2, 1, 3, 0)
tfr_alpha_old_arr = np.stack(tfr_alpha_old_list, axis=2).transpose(2, 1, 3, 0)
X_h2c = [tfr_alpha_new_arr, tfr_alpha_old_arr]
# %%
# Make sensor-frequency adjacancy matrix for alpha channels
sensor_adjacency_alpha, ch_names_alpha = find_ch_adjacency(
    epochs_complete[1].copy().pick_channels(ch_posterior).info, "eeg"
)
tf_timepoints = tfr_alpha_new_arr.shape[2]
tfr_adjacency_alpha = mne.stats.combine_adjacency(
    len(alpha_freqs), tf_timepoints, sensor_adjacency_alpha
)
# %%
# Calculate statistical thresholds, h2c
t_obs_h2c, clusters_h2c, cluster_pv_h2c, h0_h2c = spatio_temporal_cluster_test(
    X_h2c, tfce, n_permutations=1000, adjacency=tfr_adjacency_alpha
)
significant_points_h2c = np.where(cluster_pv_h2c < p_accept)[0]
# %%
# calculate power difference
grand_avg_tfr_alpha_old = np.average(tfr_alpha_old_arr, axis=0)
grand_avg_tfr_alpha_new = np.average(tfr_alpha_new_arr, axis=0)
tfr_alpha_diff = np.subtract(
    grand_avg_tfr_alpha_old, grand_avg_tfr_alpha_new
).transpose(1, 0, 2)
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
# save report
report.save(fname_report, overwrite=overwrite)
