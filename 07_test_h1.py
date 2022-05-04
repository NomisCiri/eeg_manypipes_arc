"""Test the first hypothesis.

> There is an effect of scene category
> (i.e., a difference between images showing man-made vs. natural environments)
> on the amplitude of the N1 component,
> i.e. the first major negative EEG voltage deflection.

"""

# %%
# Imports

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.stats as stats
import seaborn as sns
from tqdm.auto import tqdm

from config import FNAME_EPO_CLEAN_TEMPLATE, FNAME_REPORT_H1, SUBJS

# %%
# Filepaths and settings

# epochs as loaded are from -1.5 to 2.5, but this is too long for this ERP analysis
# 1s (or up to 3s) prior to 0s is ITI, 0s to 0.5s is image presentation, then response,
# then feedback
crop = (-0.2, 0.5)

# epochs as loaded are not yet baseline corrected, do it now
baseline = (None, 0)

conditions = ["natural", "man_made"]

overwrite = True

# clusterperm settings
pthresh = 0.001
thresh = stats.distributions.t.ppf(1 - pthresh, len(SUBJS) - 1)
nperm = 1024
seed_H1 = 59739

# %%
# Read all epochs and make ERPs per subj

evokeds = {i: [] for i in conditions}
for sub in tqdm(SUBJS):
    fname = FNAME_EPO_CLEAN_TEMPLATE.format(sub=sub)
    epochs = mne.read_epochs(fname, preload=True, verbose=0)
    epochs.crop(*crop, verbose=0)
    epochs.apply_baseline(baseline=baseline, verbose=0)

    for condi in conditions:
        evokeds[condi] += [epochs[condi].average()]

# %%
# Start a report
report = mne.Report(title="Hypothesis 1")

# %%
# Plot topomaps for each condition

# need to form grand average
grand_ave_evokeds = {}
for condi, evoked_list in evokeds.items():
    grand_ave_evokeds[condi] = mne.grand_average(evoked_list)

# N1 is relevant from about 100ms to 200ms
times = np.arange(-0.05, 0.275, 0.05)
average = 0.05

# plot both conditions in one figure: one row per condi
fig, axs = plt.subplots(2, len(times) + 1, figsize=(8, 4))
uVmax = 9

kwargs = dict(
    times=times, average=average, ch_type="eeg", vmin=-uVmax, vmax=uVmax, show=False
)
with sns.plotting_context("talk"):
    with plt.style.context("default"):
        for icondi, condi in enumerate(conditions):
            _ = grand_ave_evokeds[condi].plot_topomap(**kwargs, axes=axs[icondi, :])

            axs[icondi, 0].set_ylabel(condi)


fig.tight_layout()
report.add_figure(fig=fig, title="Topomaps", image_format="PNG", replace=True)

# %%
# Plot sensor overview
with plt.style.context("default"):
    fig = epochs.plot_sensors(show_names=True, show=False)

report.add_figure(fig=fig, title="Sensor overview", image_format="PNG", replace=True)
# %%
# Plot timecourse fronto-central
kwargs_lineplot = dict(combine="mean", ci=0.68, show_sensors=False, show=False)
grp_frontocentral = ["AFz", "F1", "Fz", "F2", "FCz"]
fig, ax = plt.subplots()
_ = mne.viz.plot_compare_evokeds(
    evokeds, picks=grp_frontocentral, **kwargs_lineplot, axes=ax
)
axins = fig.add_axes([0.1, 0.25, 0.2, 0.2])
info = epochs.copy().pick_channels(grp_frontocentral).info
mne.viz.plot_sensors(
    info, kind="topomap", title="", sphere=100, axes=axins, show=False, pointsize=2
)

report.add_figure(
    fig=fig, title="Timecourse fronto-central", image_format="PNG", replace=True
)
# %%
# Plot timecourse central
grp_central = ["FCz", "C1", "Cz", "C2", "CPz"]
fig, ax = plt.subplots()
_ = mne.viz.plot_compare_evokeds(evokeds, picks=grp_central, **kwargs_lineplot, axes=ax)
axins = fig.add_axes([0.1, 0.25, 0.2, 0.2])
info = epochs.copy().pick_channels(grp_central).info
mne.viz.plot_sensors(
    info, kind="topomap", title="", sphere=100, axes=axins, show=False, pointsize=2
)

report.add_figure(fig=fig, title="Timecourse central", image_format="PNG", replace=True)

# %%
# Plot timecourse parieto-occipital
grp_parietooccipital = ["PO3", "POz", "PO4", "O1", "Oz", "O2"]
fig, ax = plt.subplots()
_ = mne.viz.plot_compare_evokeds(
    evokeds, picks=grp_parietooccipital, **kwargs_lineplot, axes=ax
)
axins = fig.add_axes([0.1, 0.25, 0.2, 0.2])
info = epochs.copy().pick_channels(grp_parietooccipital).info
mne.viz.plot_sensors(
    info, kind="topomap", title="", sphere=100, axes=axins, show=False, pointsize=2
)

report.add_figure(
    fig=fig, title="Timecourse parieto-occipital", image_format="PNG", replace=True
)

# %%
# Cluster permutation test

# find sensor adjacency
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(
    info=epochs.copy().info, ch_type="eeg"
)

# prepare data "X": subjects x timepoints x channels
# where in each subject we subtracted "man_made" from "natural" ERP
# (channels x timepoints)
datas = []
for evo1, evo2 in zip(evokeds["natural"], evokeds["man_made"]):
    data1 = evo1.get_data()
    data2 = evo2.get_data()
    datas += [data1 - data2]

X = np.stack(datas).transpose(0, 2, 1)

# run the test
t_obs, clusters, cluster_pv, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
    X=X,
    threshold=thresh,
    adjacency=sensor_adjacency,
    n_permutations=nperm,
    seed=seed_H1,
    out_type="mask",
)

# %%
# Visualize significant clusters
sig_clusters = np.array(clusters)[cluster_pv < pthresh]

for clu in range(sig_clusters.shape[0]):
    sig_t = np.any(sig_clusters[clu], axis=1)
    sig_ch = np.any(sig_clusters[clu], axis=0)
    grp_sig = np.array(epochs.ch_names)[sig_ch].tolist()

    fig, ax = plt.subplots()
    _ = mne.viz.plot_compare_evokeds(evokeds, picks=grp_sig, **kwargs_lineplot, axes=ax)
    ax.plot(epochs.times[sig_t], [ax.get_ylim()[0]] * np.sum(sig_t), "rs")
    axins = fig.add_axes([0.1, 0.25, 0.2, 0.2])
    info = epochs.copy().pick_channels(grp_sig).info
    mne.viz.plot_sensors(
        info, kind="topomap", title="", sphere=100, axes=axins, show=False, pointsize=2
    )
    fig

    report.add_figure(
        fig=fig, title=f"Cluster #{clu}", image_format="PNG", replace=True
    )

# %%
# Save report
report.save(FNAME_REPORT_H1, open_browser=False, overwrite=overwrite)

# %%
