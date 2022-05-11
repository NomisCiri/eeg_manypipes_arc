"""Test the hypotheses specified in the instructions.

> There are effects of successful recognition of old images
> (i.e., a difference between old images correctly recognized as old [hits] vs.
> old images incorrectly judged as new [misses]) ...
> a. ... on EEG voltage at any channels, at any time.
> b. ... on spectral power, at any frequencies, at any channels, at any time.

"""

# %%
# Imports
import itertools
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
    FNAME_HYPOTHESES_3_TEMPLATE,
    FNAME_REPORT_H3,
    FPATH_DS,
    OVERWRITE_MSG,
    SUBJS,
    TRIGGER_CODES,
)
from utils import catch, parse_overwrite

# %%
# Filepaths and settings
fpath_ds = FPATH_DS
overwrite = True
fname_report = FNAME_REPORT_H3
fname_h3a = Path(FNAME_HYPOTHESES_3_TEMPLATE.format(h="h3a_cluster.pkl"))
fname_h3b_wavelet = Path(FNAME_HYPOTHESES_3_TEMPLATE.format(h="h3b_wavelet.pkl"))
fname_h3b_cluster = Path(FNAME_HYPOTHESES_3_TEMPLATE.format(h="h3b_cluster.pkl"))

# Settings for cluster test
pthresh = 0.05  # general significance alpha level
pthresh_cluster = 0.001  # cluster forming alpha level
tail = 0  # two-tailed, see also "pthresh / 2" below
thresh = stats.distributions.t.ppf(1 - pthresh_cluster / 2, len(SUBJS) - 1)
sigma = 1e-3  # sigma for the small variance correction
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
nperm = 10000
seed_H3 = 42


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
        ["old"],
        ["hit"],
        list(TRIGGER_CODES[3].values()),
    )
)
# List of all trigger combinations for an old image
triggers_misses_list = list(
    itertools.product(
        list(TRIGGER_CODES[0].values()),
        ["old"],
        ["miss"],
        list(TRIGGER_CODES[3].values()),
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
# Concatanate conditions for use with cluster based permutation test
evokeds_diff_arr = np.stack(evokeds_diff_list, axis=2).transpose(2, 1, 0)

# %%
# Calculate adjacency matrix between sensors from their locations
sensor_adjacency, ch_names = find_ch_adjacency(epochs_complete[1].copy().info, "eeg")

# %%
# Calculate statistical thresholds
# Check overwrite
# If there is a cluster test, and overwrite is false, load data

if fname_h3a.exists() and not overwrite:
    with open(fname_h3a, "rb") as fin:
        clusterstats = pickle.load(fin)

# If overwriting is false compute everything again
else:
    clusterstats = spatio_temporal_cluster_1samp_test(
        evokeds_diff_arr,
        threshold=thresh,
        n_permutations=nperm,
        adjacency=sensor_adjacency,
        stat_fun=stat_fun_hat,
        tail=tail,
        seed=seed_H3,
    )
    with open(fname_h3a, "wb") as fout:
        pickle.dump(clusterstats, fout)

t_obs_h3a, clusters_h3a, cluster_pv_h3a, h0_h3a = clusterstats
sig_cluster_inds_h3a = np.where(cluster_pv_h3a < pthresh)[0]
# %%
# Hypothesis 3b.
# Do wavelet tranformation on whole epoch to get tfr
# If there is a wavelet file test, and overwrite is false, load data
if fname_h3b_wavelet.exists() and not overwrite:
    with open(fname_h3b_wavelet, "rb") as fin:
        tfr_diff_list = pickle.load(fin)
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
    with open(fname_h3b_wavelet, "wb") as fout:
        pickle.dump(tfr_diff_list, fout)

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
if fname_h3b_cluster.exists() and not overwrite:
    with open(fname_h3b_cluster, "rb") as fin:
        clusterstats_h3b = pickle.load(fin)

else:
    clusterstats = spatio_temporal_cluster_1samp_test(
        tfr_diff_arr,
        threshold=thresh,
        n_permutations=10000,
        adjacency=tfr_adjacency,
        stat_fun=stat_fun_hat,
        tail=tail,
        n_jobs=40,
        seed=seed_H3,
    )
    with open(fname_h3b_cluster, "wb") as fout:
        pickle.dump(clusterstats, fout)

t_obs_diff_h3b, clusters_diff_h3b, cluster_pv_diff_h3b, h0_diff_h3b = clusterstats
sig_cluster_inds_h3b = np.where(cluster_pv_diff_h3b < pthresh)[0]

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
