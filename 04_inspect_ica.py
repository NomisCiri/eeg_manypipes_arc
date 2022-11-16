"""Identify bad ICA components and apply cleaned solution to raw.

Either load existing metadata on which ICA components to reject,
or visually screen components and save metadata.

When running this in interactive mode, inspect the ICA components
to determine which to exclude.
Once you have decided, fix it in the code and run the code
with a re-started session to produce the files and reports.

"""

# %%
# Imports
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
from mne.preprocessing import create_eog_epochs, read_ica
from mne.utils import logger

from config import (
    FNAME_BADS_TEMPLATE,
    FNAME_COMPONENTS_TEMPLATE,
    FNAME_ICA_RAW_TEMPLATE,
    FNAME_ICA_TEMPLATE,
    FNAME_RAW_SET_TEMPLATE,
    FNAME_REPORT_ICA_TEMPLATE,
    FNAME_SEGMENTS_TEMPLATE,
    FPATH_DS,
    FPATH_DS_NOT_FOUND_MSG,
    NOTCH_FREQS,
    OVERWRITE_MSG,
    PATH_NOT_FOUND_MSG,
    SUBJS,
)
from utils import get_raw_data, parse_overwrite

# %%
# Filepaths and settings

sub = 1
overwrite = True
interactive = True

# Whether or not to accept the automatically suggested components for exclusion
# will be set to True if an env variable "CI" is 'true'
accept_automatic = False


# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(sub=sub, overwrite=overwrite, interactive=interactive)

    defaults = parse_overwrite(defaults)

    sub = defaults["sub"]
    overwrite = defaults["overwrite"]
    interactive = defaults["interactive"]

# Check inputs after potential overwriting
if sub not in SUBJS:
    raise ValueError(f"'{sub}' is not a valid subject ID.\nUse: {SUBJS}")
if not FPATH_DS.exists():
    raise RuntimeError(FPATH_DS_NOT_FOUND_MSG.format(FPATH_DS))
if overwrite:
    logger.info("`overwrite` is set to ``True``.")

# %%
# Check overwrite
fname_comps = Path(FNAME_COMPONENTS_TEMPLATE.format(sub=sub))
if fname_comps.exists() and interactive and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_comps))

fname_icaraw = Path(FNAME_ICA_RAW_TEMPLATE.format(sub=sub))
if fname_icaraw.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_icaraw))

fname_report = Path(FNAME_REPORT_ICA_TEMPLATE.format(sub=sub))
if fname_report.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_report))

# %%
# Start a report to save figures
report = mne.Report(title=f"ICA, subject {sub}")

# %%
# Get raw data
raw = get_raw_data(FNAME_RAW_SET_TEMPLATE.format(sub=sub))
raw.load_data()

# %%
# Prepare unprocessed copy of raw to apply ICA solution to
raw_copy = raw.copy()

# %%
# Set bad channels
fname_bads = Path(FNAME_BADS_TEMPLATE.format(sub=sub))
if fname_bads.exists():
    with open(fname_bads, "r") as fin:
        bads_dict = json.load(fin)

    bads = []
    for _, chs in bads_dict.items():
        bads += chs
    bads = list(set(bads))
    raw.info["bads"] = bads

else:
    logger.info("File on bad channels not found. Did you run 01_find_bads.py?")

# %%
# Set bad segments
fname_seg = Path(FNAME_SEGMENTS_TEMPLATE.format(sub=sub))
if fname_seg.exists():
    annots_bad = mne.read_annotations(fname_seg)
    raw.set_annotations(annots_bad)
else:
    logger.info("File on bad segments not found. Did you run 02_mark_bad_segments.py?")

# %%
# highpass filter
raw.load_data()
raw = raw.filter(l_freq=1.0, h_freq=None)

# %%
# Downsample
raw = raw.resample(sfreq=128)

# %%
# Get ICA solution
fname_ica = Path(FNAME_ICA_TEMPLATE.format(sub=sub))
ica = read_ica(fname_ica)

# %%
# If not interactive, exclude components and exit
if not interactive:

    # file with components to exclude must exist
    if not fname_comps.exists():
        raise RuntimeError(PATH_NOT_FOUND_MSG.format(fname_comps))

    with open(fname_comps, "r") as fin:
        comps_dict = json.load(fin)

    ica.exclude = comps_dict["exclude"]
    raw_clean = ica.apply(inst=raw_copy)
    raw_clean.save(fname_icaraw, overwrite=overwrite)
    sys.exit()

# %%
# Else: interactive run! ... screen ICA components
CI = os.getenv("CI")
if CI == "true":
    if not accept_automatic:
        logger.info("Overriding `accept_automatic` through environment variable 'CI'.")
    accept_automatic = True

if accept_automatic:
    msg = (
        "Currently accepting automatically identified components for exclusion. "
        "Set `accept_automatic = False` for visual screening. "
        "Check for environment variables named 'CI' and make sure they are not 'true'"
    )
    logger.info(msg)

# %%
# Automatically find artifact components using the EOG data
veog_idx, veog_scores = ica.find_bads_eog(raw, "VEOG")
heog_idx, heog_scores = ica.find_bads_eog(raw, "HEOG")

# %%
# If we accept automatic, quit early - but don't save excluded indices
if accept_automatic:
    ica.exclude = list(set(veog_idx + heog_idx))
    raw_clean = ica.apply(inst=raw_copy)
    raw_clean.save(fname_icaraw, overwrite=overwrite)
    sys.exit()

# %%
# Plot components
with plt.style.context("default"):
    fig = ica.plot_components(inst=raw)

report.add_figure(fig=fig, title="ICA components", image_format="PNG")

# %%
# Create VEOG epochs and plot evoked
epochs_veog = create_eog_epochs(raw, ch_name="VEOG", picks="eeg")
fig = epochs_veog.average().plot()

report.add_figure(fig=fig, title="VEOG Epochs", image_format="PNG")


# %%
# Create HEOG epochs and plot evoked
epochs_heog = create_eog_epochs(raw, ch_name="HEOG", picks="eeg")
fig = epochs_heog.average().plot()

report.add_figure(fig=fig, title="HEOG Epochs", image_format="PNG")

# %%
# Plot VEOG scores and overlay
exclude_veog = veog_idx  # change me! (use a list of component indices)
fig = ica.plot_scores(
    veog_scores, exclude=exclude_veog, title=f"VEOG, exclude: {exclude_veog}"
)

report.add_figure(fig=fig, title="VEOG scores", image_format="PNG")

fig = ica.plot_overlay(epochs_veog.average(), exclude=exclude_veog, show=False)
fig.tight_layout()

report.add_figure(fig=fig, title="VEOG overlay", image_format="PNG")

# %%
# Plot HEOG scores and overlay
exclude_heog = heog_idx  # change me! (use a list of component indices)
fig = ica.plot_scores(
    heog_scores, exclude=exclude_heog, title=f"HEOG, exclude: {exclude_heog}"
)

report.add_figure(fig=fig, title="HEOG scores", image_format="PNG")

fig = ica.plot_overlay(epochs_heog.average(), exclude=exclude_heog, show=False)
fig.tight_layout()

report.add_figure(fig=fig, title="HEOG overlay", image_format="PNG")

# %%
# Set ica.exclude attribute
ica.exclude = sorted(list(set(exclude_veog + exclude_heog)))
exclude = [int(i) for i in ica.exclude]  # convert numpy.int64 -> int
assert exclude == ica.exclude
print(f"Excluding: {exclude}")

with open(fname_comps, "w") as fout:
    json.dump(dict(exclude=sorted(exclude)), fout, indent=4, sort_keys=True)
    fout.write("\n")

# %%
# Apply ICA to raw data
# Exclude components from ica.exclude
raw_clean = ica.apply(inst=raw_copy)

# %%
# Visually screen data for absence of blinks and saccades
# first remove line noise (on a copy)
raw_clean_copy = raw_clean.copy()
raw_clean_copy.notch_filter(NOTCH_FREQS)
raw_clean_copy.plot(
    block=True,
    use_opengl=False,
    n_channels=len(raw_clean.ch_names),
    bad_color="red",
    duration=20.0,
    clipping=None,
)

# %%
# Save as ica-cleaned data
raw_clean.save(fname_icaraw, overwrite=overwrite)

# %%
# Save report
report.save(fname_report, open_browser=False, overwrite=overwrite)

# %%
