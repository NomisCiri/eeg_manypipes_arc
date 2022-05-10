"""Test the hypotheses specified in the instructions.

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

import mne
import numpy as np
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test, ttest_1samp_no_p
from mne.time_frequency import tfr_morlet
from scipy import stats

from config import (
    FNAME_HYPOTHESES_4_TEMPLATE,
    FNAME_REPORT_H4,
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
fname_report = FNAME_REPORT_H4
fname_h4a = Path(FNAME_HYPOTHESES_4_TEMPLATE.format(h="h4a_cluster.pkl"))
fname_h4b_wavelet = Path(FNAME_HYPOTHESES_4_TEMPLATE.format(h="h4b_wavelet.pkl"))
fname_h4b_cluster = Path(FNAME_HYPOTHESES_4_TEMPLATE.format(h="h4b_cluster.pkl"))

# Settings for cluster test
p_accept = 0.001
sigma = 1e-3  # sigma for the "hat" method
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
threshold = stats.distributions.t.ppf(1 - p_accept, len(SUBJS) - 1)  # threshold
seed_H3 = 1991
nperm = 10000
tail = 0
# Time frequency
freqs = np.logspace(*np.log10([4, 100]), num=40).round()
n_cycles = freqs / 2.0  # different number of cycle per frequency
n_cycles.round()
# toi
toi_min = -0.2
toi_max = 1
# List of all trigger combinations for a new image
triggers_hits_list = list(
    itertools.product(
        list(TRIGGER_CODES[0].values()),
        list(TRIGGER_CODES[1].values()),
        list(TRIGGER_CODES[2].values()),
        "sub_remembered",
    )
)
# List of all trigger combinations for an old image
triggers_misses_list = list(
    itertools.product(
        list(TRIGGER_CODES[0].values()),
        list(TRIGGER_CODES[1].values()),
        list(TRIGGER_CODES[2].values()),
        "sub_forgotten",
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
            x[triggers_hits]
            .crop(toi_min, toi_max)
            .apply_baseline(None, 0)
            .average()
            .get_data(),
            x[triggers_misses]
            .crop(toi_min, toi_max)
            .apply_baseline(None, 0)
            .average()
            .get_data(),
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
        n_permutations=nperm,
        adjacency=sensor_adjacency,
        n_jobs=40,
        stat_fun=stat_fun_hat,
        tail=tail,
    )
    file = open(fname_h4a, "wb")
    pickle.dump(clusterstats, file)
    file.close()

t_obs_h4a, clusters_h4a, cluster_pv_h4a, h0_h4a = clusterstats
sig_cluster_inds_h4a = np.where(cluster_pv_h4a < 0.01)[0]
# %%
# Hypothesis 3b.
# Do wavelet tranformation on whole epoch to get tfr
# If there is a wavelet file test, and overwrite is false, load data
if fname_h4b_wavelet.exists() and not overwrite:
    file_wavelet = open(fname_h4b_wavelet, "rb")
    tfr_diff_list = pickle.load(file_wavelet)
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
        n_permutations=nperm,
        adjacency=tfr_adjacency,
        stat_fun=stat_fun_hat,
        tail=tail,
    )
    file_h4b_cluster = open(fname_h4b_cluster, "wb")
    pickle.dump(clusterstats, file_h4b_cluster)
    file_h4b_cluster.close()

t_obs_diff_h4b, clusters_diff_h4b, cluster_pv_diff_h4b, h0_diff_h4b = clusterstats
sig_cluster_inds_h4b = np.where(cluster_pv_diff_h4b < 0.01)[0]

# %%
# unpack cluster statistics
# make dummy tfr for figure
tfr_specs_dummy = tfr_morlet(
    epochs_complete[0][triggers_hits],
    freqs,
    n_cycles=n_cycles,
    average=True,
    return_itc=False,
    n_jobs=6,
).crop(toi_min, toi_max)

# %%
# get cluster info
times_min_h4b = list()
times_max_h4b = list()
channels_h4b = list()
freqs_h4b = list()

for i in range(0, len(sig_cluster_inds_h4b)):
    times_min_h4b.append(
        tfr_specs_dummy.times[min(clusters_diff_h4b[sig_cluster_inds_h4b[i]][1])]
    )
    times_max_h4b.append(
        tfr_specs_dummy.times[max(clusters_diff_h4b[sig_cluster_inds_h4b[i]][1])]
    )
    channels_h4b.append(
        np.array(tfr_specs_dummy.ch_names)[
            np.unique(clusters_diff_h4b[sig_cluster_inds_h4b[i]][2])
        ]
    )
    freqs_h4b.append(
        tfr_specs_dummy.freqs[np.unique(clusters_diff_h4b[sig_cluster_inds_h4b[i]][0])]
    )
# %%
