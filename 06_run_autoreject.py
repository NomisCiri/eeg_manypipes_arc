"""Apply autoreject to the epochs and apply average reference to channels.

1. Load epochs
2. Apply autoreject (fit and transform)
3. Save report
4. Apply average reference
5. Save clean epochs
"""

# %%
# Imports
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from autoreject import AutoReject
from mne.utils import logger
from psutil import virtual_memory

from config import (
    FNAME_AR_OBJECT_TEMPLATE,
    FNAME_AR_PLOT_TEMPLATE,
    FNAME_AR_REJECT_LOG_TEMPLATE,
    FNAME_BAD_EPOS_TEMPLATE,
    FNAME_EPO_CLEAN_TEMPLATE,
    FNAME_EPOCHS_TEMPLATE,
    FPATH_DS,
    OVERWRITE_MSG,
    PATH_NOT_FOUND_MSG,
    REF_CHANNEL,
    SUBJS,
)
from utils import parse_overwrite

# %%
# Filepaths and settings

sub = 1
fpath_ds = FPATH_DS
overwrite = True

rng_autoreject = 31415


# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        sub=sub,
        fpath_ds=fpath_ds,
        overwrite=overwrite,
    )

    defaults = parse_overwrite(defaults)

    sub = defaults["sub"]
    fpath_ds = defaults["fpath_ds"]
    overwrite = defaults["overwrite"]

# Check inputs after potential overwriting
if sub not in SUBJS:
    raise ValueError(f"'{sub}' is not a valid subject ID.\nUse: {SUBJS}")
if not fpath_ds.exists():
    raise RuntimeError(PATH_NOT_FOUND_MSG.format(fpath_ds))
if overwrite:
    logger.info("`overwrite` is set to ``True``.")

# %%
# Check overwrite
fname_epo_clean = Path(FNAME_EPO_CLEAN_TEMPLATE.format(sub=sub))
if fname_epo_clean.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_epo_clean))

fname_ar_object = Path(FNAME_AR_OBJECT_TEMPLATE.format(sub=sub))
if fname_ar_object.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_ar_object))

fname_ar_reject_log = Path(FNAME_AR_REJECT_LOG_TEMPLATE.format(sub=sub))
if fname_ar_reject_log.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_ar_reject_log))

fname_ar_plot = Path(FNAME_AR_PLOT_TEMPLATE.format(sub=sub))
if fname_ar_plot.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_ar_plot))

fname_bad_epos = Path(FNAME_BAD_EPOS_TEMPLATE.format(sub=sub))
if fname_bad_epos.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_bad_epos))

# %%
# Load "dirty" epochs
fname_epochs = Path(FNAME_EPOCHS_TEMPLATE.format(sub=sub))
epochs = mne.read_epochs(fname_epochs)

# %%
# Apply autoreject
available_gb = virtual_memory().available * 1e-9
n_jobs = 1
if available_gb > 8:
    n_jobs = 6
ar = AutoReject(
    n_interpolate=[2, 4, 8],
    consensus=np.arange(0.2, 0.8, 0.1),
    n_jobs=n_jobs,
    random_state=rng_autoreject,
)

epochs.info["bads"] = [REF_CHANNEL]
epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

# %%
# Save autoreject files
ar.save(fname_ar_object, overwrite=overwrite)
# reject_log.save(fname_ar_reject_log, overwrite=overwrite)  # only autoreject>0.3.1


# %%
# Plot reject log and save
with sns.plotting_context("talk"):
    fig, ax = plt.subplots(figsize=(30, 20))
    reject_log.plot("horizontal", ax=ax)

fig.savefig(fname_ar_plot, bbox_inches="tight")

# %%
# Save bad epochs
bad_epochs_dict = {}
bad_epochs_dict["bad_epochs_idxs_0-based"] = np.arange(1200)[
    reject_log.bad_epochs
].tolist()

with open(fname_bad_epos, "w") as fout:
    json.dump(bad_epochs_dict, fout, indent=4, sort_keys=True)
    fout.write("\n")

# %%
# Re-reference to common average reference
epochs_clean.info["bads"] = []  # interpolate original ref channel
epochs_clean.set_eeg_reference(ref_channels="average", projection=False, ch_type="eeg")

# %%
# Save clean epochs
epochs_clean.save(fname_epo_clean, overwrite=overwrite)

# %%
