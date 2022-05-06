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
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.auto import tqdm

from config import FNAME_EPO_CLEAN_TEMPLATE, FNAME_REPORT_H1, SUBJS

# %%
# Filepaths and settings

# epochs as loaded are from -1.5 to 2.5, but this is too long for this ERP analysis
# 1s (or up to 3s) prior to 0s is ITI, 0s to 0.5s is image presentation, then response,
# then feedback
crop = (-0.1, 0.2)

# epochs as loaded are not yet baseline corrected, do it now
baseline = (None, 0)

conditions = ["natural", "man_made"]

overwrite = True

window_n1 = (0.08, 0.15)  # approximately

# clusterperm settings
pthresh = 0.05  # general significance alpha level
pthresh_cluster = 0.001  # cluster forming alpha level
tail = 0  # two-tailed, see also "pthresh / 2" below
thresh = stats.distributions.t.ppf(1 - pthresh_cluster / 2, len(SUBJS) - 1)
nperm = 5000
seed_H1 = 59739
# Run test only over frontal to centro-parietal channels
ch_exclude_permtest = [
    "Afp9",
    "Afp10",
    "FT7",
    "FT8",
    "T7",
    "T8",
    "M1",
    "M2",
    "TP7",
    "TP8",
    "CP5",
    "CP6",
    "P9",
    "P7",
    "P5",
    "P3",
    "P10",
    "P8",
    "P6",
    "P4",
    "PO7",
    "PO3",
    "POz",
    "PO8",
    "PO4",
    "O1",
    "Oz",
    "O2",
    "Iz",
]

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
# Add description text to report
html = """
<p>There is an effect of scene category
(i.e., a difference between images showing man-made vs. natural environments)
on the amplitude of the N1 component,
i.e. the first major negative EEG voltage deflection.</p>
"""

report.add_html(title="Hypothesis", html=html, replace=True)

html = f"""
<p>The following processing steps are applied (per subject):</p>
<ol>
<li>Epochs are loaded from disk (centered on image onset)</li>
<li>Epochs are cropped to the range: {crop} (in seconds)</li>
<li>Epochs are baseline corrected using: {baseline}
("None" stands for first/last sample)</li>
<li>Epochs are averaged to ERPs with conditions
&quot;man_made&quot; and &quot;natural&quot;</li>
</ol>
<p>Below we show several plots and tests of the data.</p>
"""

report.add_html(title="Data", html=html, replace=True)

# %%
# Plot topomaps for each condition

# need to form grand average
grand_ave_evokeds = {}
for condi, evoked_list in evokeds.items():
    grand_ave_evokeds[condi] = mne.grand_average(evoked_list)

# N1 is relevant from about 100ms to 200ms
times = np.arange(0.0, 0.180, 0.025)
average = 0.01

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
    fig=fig,
    title="Timecourse fronto-central group",
    image_format="PNG",
    replace=True,
    caption="Shading = SEM",
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

report.add_figure(
    fig=fig,
    title="Timecourse central group",
    image_format="PNG",
    replace=True,
    caption="Shading = SEM",
)

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
    fig=fig,
    title="Timecourse parieto-occipital group",
    image_format="PNG",
    replace=True,
    caption="Shading = SEM",
)

# %%
# Cluster permutation test

# find sensor adjacency
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(
    info=epochs.copy().info, ch_type="eeg"
)

# Plot sensor overview, marking those used in clusterperm
ch_permtest = list(set(ch_names) - set(ch_exclude_permtest))
epo_c = epochs.copy()
epo_c.info["bads"] = ch_permtest
fig, ax = plt.subplots()
fig = epo_c.plot_sensors(show_names=True, show=False, axes=ax)

caption = (
    "We exclude lateral and occipital sensors, "
    "because images were presented centrally, "
    "and the topography suggests a frontal / central distribution for the N1."
)

report.add_figure(
    fig=fig,
    title="Sensor overview (those included in cluster-permutation test are marked red)",
    image_format="PNG",
    replace=True,
    caption=caption,
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

spatial_exclude = [ch_names.index(i) for i in ch_exclude_permtest]

# run the test
t_obs, clusters, cluster_pv, _ = mne.stats.spatio_temporal_cluster_1samp_test(
    X=X,
    threshold=thresh,
    tail=tail,
    adjacency=sensor_adjacency,
    n_permutations=nperm,
    seed=seed_H1,
    out_type="mask",
    spatial_exclude=spatial_exclude,
)

# %%
# Plot channel values "natural" - "man_made"
nrows = int(np.sqrt(len(ch_permtest)))
fig, axs = plt.subplots(
    nrows,
    nrows + 1,
    figsize=(nrows * 2, nrows * 2),
    sharex=True,
    sharey=True,
)

for iax, ax in enumerate(axs.flat):
    if iax >= len(ch_permtest):
        ax.remove()
        continue

    ch = ch_permtest[iax]
    ich = ch_names.index(ch)

    df = pd.DataFrame(X[..., ich] * 1e6).T
    df["times"] = epochs.times
    df = df.melt(id_vars=["times"], var_name="subject", value_name="µV")

    sns.lineplot(x="times", y="µV", data=df, ci=68, n_boot=100, ax=ax)

    title = ax.set_title(ch_names[ich])

    if ch in grp_frontocentral:
        title.set_color("blue")
    elif ch in grp_central:
        title.set_color("magenta")
    elif ch in grp_parietooccipital:
        title.set_color("green")

    ax.axhline(0, color="black", lw=1, ls="--")
    ax.axvline(0, color="black", lw=1, ls="--")

    ax.axvspan(*window_n1, color="black", alpha=0.1)

    ys = -2.0 * (np.abs(t_obs) > thresh)[..., ich].astype(int)
    ys[ys == 0] = np.nan
    ax.plot(epochs.times, ys, "ro")

sns.despine(fig)
fig.tight_layout()

title = (
    "Difference waves: 'natural' - 'man_made' per channel "
    "(red bar shows significance)"
)
caption = (
    "Only shown for channels used in cluster-permutation testing.\n"
    f"Shading = SEM; Significance level p={pthresh} (uncorrected).\n"
    f"Gray window marks approximate N1 window: {window_n1} (in seconds).\n"
    "Blue channels: fronto-central group; Magenta channels: central group."
)
report.add_figure(
    fig=fig,
    title=title,
    image_format="PNG",
    replace=True,
    caption=caption,
)


# %%
# Visualize significant clusters
iclu_sig = 0
for iclu in range(len(clusters)):

    # only viz significant clusters
    pval = cluster_pv[iclu]
    if pval >= pthresh:
        continue
    else:
        iclu_sig += 1
        # we plot

    sig_t = np.any(np.array(clusters)[iclu], axis=1)
    sig_ch = np.any(np.array(clusters)[iclu], axis=0)
    grp_sig = np.array(epochs.ch_names)[sig_ch].tolist()

    fig, ax = plt.subplots()
    fig.tight_layout()
    _ = mne.viz.plot_compare_evokeds(evokeds, picks=grp_sig, **kwargs_lineplot, axes=ax)
    ax.plot(epochs.times[sig_t], [ax.get_ylim()[0]] * np.sum(sig_t), "rs")
    ax.axvspan(*window_n1, color="black", alpha=0.1)
    axins = fig.add_axes([0.75, 0.75, 0.2, 0.2])

    # Get t-value topography
    t_map = t_obs[sig_t].mean(axis=0)
    mask = np.zeros((t_map.shape[0], 1), dtype=bool)
    mask[sig_ch, :] = True

    # plot t-value topo
    t_evoked = mne.EvokedArray(t_map[:, np.newaxis], epochs.info, tmin=0)
    t_evoked.plot_topomap(
        times=0,
        mask=mask,
        axes=axins,
        cmap="Reds",
        vmin=np.min,
        vmax=np.max,
        show=False,
        colorbar=False,
        mask_params=dict(markersize=10),
    )

    # add axes for colorbar
    divider = make_axes_locatable(axins)
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
    image = axins.images[0]
    plt.colorbar(image, cax=ax_colorbar)
    xlabel = (
        f"Averaged t-map\n({epochs.times[sig_t].min():0.3f} "
        f"- {epochs.times[sig_t].max():0.3f} s)"
    )
    axins.set_xlabel(xlabel)
    axins.set_title("")

    title = (
        f"Cluster #{iclu_sig}, p={pval:04} "
        f"(cluster-forming threshold={pthresh_cluster}, two-tailed)"
    )
    report.add_figure(
        fig=fig,
        title=title,
        image_format="PNG",
        replace=True,
        caption="Shading = SEM; Gray window marks approximate N1 window",
    )

# %%
# Save report
report.save(FNAME_REPORT_H1, open_browser=False, overwrite=overwrite)

# %%
