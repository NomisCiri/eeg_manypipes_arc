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
from utils import catch  # , parse_overwrite

# %%
# Path and settings
fpath_ds = FPATH_DS
overwrite = True
# Settings for cluster test
tfce = dict(start=0.1, step=0.05)
# Time frequency
n_cycles = 7
alpha_freqs = np.arange(8, 12.5, 0.5)  # define frequencies of interest
theta_freqs = np.arange(4, 7.5, 0.5)  # define frequencies of interest

# Roi & toi
ch_fronto_central = ["FCz", "FC3", "FC4", "Fz", "FC1", "FC2"]
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
sensor_adjacency, ch_names = find_ch_adjacency(
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
evoked.plot_joint(
    title="Old - New images",
    ts_args=time_unit,
    times=[0.3, 0.35, 0.4, 0.45, 0.5],
    topomap_args=time_unit,
)
# Create ROIs by checking channel labels
# only check tois
# Visualize the results
toi_evoked = evoked.copy().crop(toi_min, toi_max)
toi_evoked.plot_image(
    colorbar=False,
    show=False,
    mask=significant_points_h2a,
    show_names="all",
    titles=None,
    **time_unit,
)
# %%
# Hypothesis 2b.
# Do wavelet tranformation on whole epoch to get tfr
tfr_alpha_new_list = list(
    [
        tfr_morlet(
            x[triggers_new].pick_channels(ch_fronto_central),
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
            x[triggers_old].pick_channels(ch_fronto_central),
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
# required format: (n_observations (subs), time,freq, n_vertices (channels)).
tfr_alpha_new_arr = np.stack(tfr_alpha_new_list, axis=2).transpose(2, 3, 1, 0)
tfr_alpha_old_arr = np.stack(tfr_alpha_old_list, axis=2).transpose(2, 3, 1, 0)
X_h2b = [tfr_alpha_new_arr, tfr_alpha_old_arr]

# %%
# Make sensor-frequency adjacancy matrix
tf_timepoints = tfr_alpha_new_arr.shape[1]
tf_adjacency = mne.stats.combine_adjacency(
    sensor_adjacency, len(alpha_freqs), tf_timepoints
)
# %%
# Calculate statistical thresholds, h2a not confirmed
t_obs_h2b, clusters_h2b, cluster_pv_h2b, h0_h2b = spatio_temporal_cluster_test(
    X_h2b, tfce, n_permutations=1000, adjacency=tf_adjacency
)
significant_points_h2b = cluster_pv_h2b.reshape(t_obs_h2b.shape).T < 0.05
# %%
# visualize results
# %%
# Hypothesis 2c.
# Do wavelet tranformation on whole epoch to get tfr
tfr_theta_new_list = list(
    [
        tfr_morlet(
            x[triggers_new].pick_channels(ch_posterior),
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
            x[triggers_old].pick_channels(ch_posterior),
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
# required format: (n_observations (subs), time,freq, n_vertices (channels)).
tfr_theta_new_arr = np.stack(tfr_theta_new_list, axis=2).transpose(2, 3, 1, 0)
tfr_theta_old_arr = np.stack(tfr_theta_old_list, axis=2).transpose(2, 3, 1, 0)
X_h2c = [tfr_theta_new_arr, tfr_theta_old_arr]

# %%
# Calculate statistical thresholds, h2a not confirmed
t_obs_h2b, clusters_h2b, cluster_pv_h2b, h0_h2b = spatio_temporal_cluster_test(
    X_h2b, tfce, n_permutations=1000, adjacency=tf_adjacency
)
significant_points_h2b = cluster_pv_h2b.reshape(t_obs_h2b.shape).T < 0.05
