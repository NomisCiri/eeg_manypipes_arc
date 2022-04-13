"""finds bad subjects based on the senstivity index dprime in a signal detection task and saves them as json"""
# %%
# Imports
import json
import sys
from pathlib import Path
from scipy import stats

from mne.utils import logger

from config import (
    FNAME_BADS_TEMPLATE,
    FPATH_DS,
    OVERWRITE_MSG,
    PATH_NOT_FOUND_MSG,
    SUBJS,
)

#%% settings
from utils import get_behavioral_data, parse_overwrite

# %%
# Filepaths and settings

sub = 1
fpath_ds = FPATH_DS
overwrite = False


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

exclude=list()# placeholder for excluding subjects
for sub in SUBJS:
    fpath_set = fpath_ds / "sourcedata" / "events" / f"EMP{sub:02}_events.csv"
    behavior_dat = get_behavioral_data(fpath_set)
    


# %%get dprime

