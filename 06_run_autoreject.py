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

import mne
from autoreject import AutoReject
from mne.utils import logger

from config import (
    FNAME_BADS_TEMPLATE,
    FNAME_EPO_CLEAN_TEMPLATE,
    FNAME_EPOCHS_TEMPLATE,
    FPATH_DS,
    OVERWRITE_MSG,
    PATH_NOT_FOUND_MSG,
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

# %%
# Load "dirty" epochs
fname_epochs = Path(FNAME_EPOCHS_TEMPLATE.format(sub=sub))
epochs = mne.read_epochs(fname_epochs)

# add bad channels (NOTE: is this needed?)
fname_bads = Path(FNAME_BADS_TEMPLATE.format(sub=sub))
if fname_bads.exists():
    with open(fname_bads, "r") as fin:
        bads_dict = json.load(fin)

    bads = []
    for _, chs in bads_dict.items():
        bads += chs
    bads = list(set(bads))
    epochs.info["bads"] = bads

else:
    logger.info("File on bad channels not found. Did you run 01_find_bads.py?")


# %%
# Apply autoreject
ar = AutoReject(n_jobs=6, random_state=rng_autoreject)

epochs_clean, reject_log = ar.fit_transform(epochs)

# %%

# %%
