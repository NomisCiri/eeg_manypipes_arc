"""Test the hypotheses specified in the instructions.

Hypotheses read:
3. There are effects of successful recognition of old images (i.e., a difference between
old images correctly recognized as old [hits] vs. old images incorrectly judged as new
[misses]) ...
a. ... on EEG voltage at any channels, at any time.
b. ... on spectral power, at any frequencies, at any channels, at any time.
"""


import itertools

# %%
# Imports
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test, spatio_temporal_cluster_test
from mne.time_frequency import tfr_morlet
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import (
    FNAME_REPORT_HYPOTHESES_TEMPLATE,
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
# Settings for cluster test
tfce = dict(start=0, step=0.2)
p_accept = 0.05
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
fname_report = Path(FNAME_REPORT_HYPOTHESES_TEMPLATE.format(h="h2"))
if fname_report.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_report))
# %%
# Start a report to save figures
report = mne.Report(title="Hypotheses 3")
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
evokeds_hits_list = list(
    [
        x[triggers_hits].crop(toi_min, toi_max).average().get_data()
        for x in epochs_complete
    ]
)
# new images
evokeds_misses_list = list(
    [
        x[triggers_misses].crop(toi_min, toi_max).average().get_data()
        for x in epochs_complete
    ]
)

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
evokeds_hits_arr = np.stack(evokeds_hits_list, axis=2).transpose(2, 1, 0)
evokeds_misses_arr = np.stack(evokeds_misses_list, axis=2).transpose(2, 1, 0)
evokeds_diff_arr = np.stack(evokeds_diff_list, axis=2).transpose(2, 1, 0)
# Concatanate conditions for use with cluster based permutation test
X_h3a = [evokeds_hits_arr, evokeds_misses_arr]
# %%
# Calculate adjacency matrix between sensors from their locations
sensor_adjacency, ch_names_theta = find_ch_adjacency(
    epochs_complete[1].copy().info, "eeg"
)
# %%
# Calculate statistical thresholds, h3a confirmed
# Check overwrite
fname_h3a = Path(FNAME_REPORT_HYPOTHESES_TEMPLATE.format(h="h3a_cluster"))
# If there is a cluster test, and overwrite is false, load data
if fname_h3a.exists() and not overwrite:
    file = open(fname_h3a, "rb")
    clusterstats = pickle.load(file)
    file.close()
# If overwriting is false compute everything again
else:
    clusterstats = spatio_temporal_cluster_test(
        X_h3a, tfce, n_permutations=1000, adjacency=sensor_adjacency, n_jobs=40
    )
    file = open(fname_h3a, "wb")
    pickle.dump(clusterstats, file)
    file.close()
    t_obs_h3a, clusters_h3a, cluster_pv_h3a, h0_h3a = clusterstats

significant_points_h3a = cluster_pv_h3a.reshape(t_obs_h3a.shape).T < 0.05

# %%
# Calculate thresholdes on within subject difference
(
    t_obs_h3a_diff,
    clusters_h3a_diff,
    cluster_pv_h3a_diff,
    h0_h3a_diff,
) = spatio_temporal_cluster_1samp_test(
    evokeds_diff_arr, tfce, n_permutations=1000, adjacency=sensor_adjacency, n_jobs=40
)
significant_points_h3a_diff = cluster_pv_h3a_diff.reshape(t_obs_h3a_diff.shape).T < 0.05
# %%
# Visualize the voltage, taking the average of all subjects
# old images
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
    title="h3a",
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
h3a_test = toi_evoked.plot_image(
    colorbar=False,
    show=False,
    mask=significant_points_h3a_diff,
    show_names="all",
    titles="Significant timepoints",
    **time_unit,
)
h3a_test.set_figheight(15)
report.add_figure(
    fig=h3a_test,
    title="h3a sig",
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
fname_h3b_wavelet = Path(FNAME_REPORT_HYPOTHESES_TEMPLATE.format(h="h3b_wavelet"))
# If there is a cluster test, and overwrite is false, load data
if fname_h3b_wavelet.exists() and not overwrite:
    file_wavelet = open(fname_h3b_wavelet, "rb")
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
    file = open(fname_h3b_wavelet, "wb")
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
fname_h3b_cluster = Path(FNAME_REPORT_HYPOTHESES_TEMPLATE.format(h="h3b_cluster"))
# If there is a cluster test filse, and overwrite is false, load data
if fname_h3b_cluster.exists() and not overwrite:
    file_cluster = open(fname_h3b_cluster, "rb")
    clusterstats_h3b = pickle.load(file)
    file.close()
else:
    clusterstats = spatio_temporal_cluster_1samp_test(
        tfr_diff_arr,
        threshold=tfce,
        n_permutations=1000,
        adjacency=tfr_adjacency,
        n_jobs=40,
    )
    t_obs_diff_h3b, clusters_diff_h3b, cluster_pv_diff_h3b, h0_diff_h3b = clusterstats
    file_cluster = open(fname_h3b_cluster, "wb")
    pickle.dump(clusterstats, file)
    file.close()
significant_points_diff_h3b = np.where(cluster_pv_diff_h3b < 0.05)[0]
# %%
# calculate average power difference
tfr_theta_diff = np.average(tfr_diff_arr, axis=0).transpose(1, 0, 2)
t_obs_diff_h3b_t = t_obs_diff_h3b.transpose(1, 0, 2)
# %%
# make h3b figure
h3b_test, axs = plt.subplots(
    nrows=len(ch_names_theta), ncols=2, figsize=(100, 20), constrained_layout=True
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
        t_obs_diff_h3b_t[:, :, ch_idx],
        aspect="auto",
        origin="lower",
        extent=[toi_min, toi_max, freqs[0], freqs[-1]],
    )
    plt.colorbar()
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Cluster T_val difference new -old \n ({ch_names_theta[ch_idx]})")


report.add_figure(
    fig=h3b_test,
    title="h2a sig",
    caption="This figure shows where the difference in theta"
    + "power between old and new images. The first column shows "
    + "The first column shows raw power  difference, the second "
    + "show the corresponding T-statistic",
    image_format="PNG",
)

# %%
for i_clu, clu_idx in enumerate(significant_points_diff_h3b):
    # unpack cluster information, get unique indices
    freq_inds, time_inds, space_inds = clusters_diff_h3b[clu_idx]
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)
    freq_inds = np.unique(freq_inds)

    # get topography for F stat
    t_map_h3b = t_obs_diff_h3b[freq_inds].mean(axis=0)
    t_map_h3b = t_map_h3b[time_inds].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = tfr_diff_list[0].times[time_inds]

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # create spatial mask
    mask = np.zeros((t_map_h3b.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(
        t_map_h3b[:, np.newaxis], tfr_diff_list[0].info, tmin=-0.2
    )
    f_evoked.plot_topomap(
        times=0,
        mask=mask,
        axes=ax_topo,
        cmap="Reds",
        vmin=np.min,
        vmax=np.max,
        show=False,
        colorbar=False,
        mask_params=dict(markersize=10),
    )
    image = ax_topo.images[0]

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        "Averaged T-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
    )

    # add new axis for spectrogram
    ax_spec = divider.append_axes("right", size="300%", pad=1.2)
    title = "Cluster #{0}, {1} spectrogram".format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += " (max over channels)"
    F_obs_plot = t_obs_diff_h3b[..., ch_inds].max(axis=-1)
    F_obs_plot_sig = np.zeros(F_obs_plot.shape) * np.nan
    F_obs_plot_sig[tuple(np.meshgrid(freq_inds, time_inds))] = F_obs_plot[
        tuple(np.meshgrid(freq_inds, time_inds))
    ]

    for f_image, cmap in zip([F_obs_plot, F_obs_plot_sig], ["gray", "autumn"]):
        c = ax_spec.imshow(
            f_image,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=[
                tfr_diff_list[0].times[0],
                tfr_diff_list[0].times[-1],
                freqs[0],
                freqs[-1],
            ],
        )
    ax_spec.set_xlabel("Time (ms)")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title(title)

    # add another colorbar
    ax_colorbar2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(c, cax=ax_colorbar2)
    ax_colorbar2.set_ylabel("F-stat")

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=0.05)
# %%
