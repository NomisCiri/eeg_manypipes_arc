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
fpath_ds = FPATH_DS
overwrite = False

# %%

exclude=list()# placeholder for excluding subjects
for sub in SUBJS:
    fpath_set = fpath_ds / "sourcedata" / "events" / f"EMP{sub:02}_events.csv"
    behavior_dat = get_behavioral_data(fpath_set)
    


# %%get dprime

