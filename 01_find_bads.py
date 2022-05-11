"""Find bad channels via PyPrep and save as JSON."""

# %%
# Imports
import json
import sys
from pathlib import Path

import pyprep
from mne.utils import logger

from config import (
    FNAME_BADS_TEMPLATE,
    FNAME_RAW_SET_TEMPLATE,
    FPATH_DS,
    FPATH_DS_NOT_FOUND_MSG,
    OVERWRITE_MSG,
    REF_CHANNEL,
    SUBJS,
)
from utils import get_raw_data, parse_overwrite

# %%
# Filepaths and settings

sub = 1
overwrite = False

pyprep_rng = 42

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        sub=sub,
        overwrite=overwrite,
    )

    defaults = parse_overwrite(defaults)

    sub = defaults["sub"]
    overwrite = defaults["overwrite"]

# Check inputs after potential overwriting
if sub not in SUBJS:
    raise ValueError(f"'{sub}' is not a valid subject ID.\nUse: {SUBJS}")
if not FPATH_DS.exists():
    raise RuntimeError(FPATH_DS_NOT_FOUND_MSG.format(FPATH_DS))
if overwrite:
    logger.info("`overwrite` is set to ``True``.")

# %%
# Check overwrite
fname_bads = Path(FNAME_BADS_TEMPLATE.format(sub=sub))
if fname_bads.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_bads))

# %%
# Get raw data
raw = get_raw_data(FNAME_RAW_SET_TEMPLATE.format(sub=sub))

# %%
# Find bad channels via pyprep
raw.load_data()
nc = pyprep.NoisyChannels(raw, random_state=pyprep_rng)
nc.find_all_bads()
bads_dict = nc.get_bads(as_dict=True)

# sanity check: POz was a reference channel, should be perfectly flat
assert REF_CHANNEL in bads_dict["bad_by_flat"]

# %%
# Save the outputs under derivatives
bads_dict_sorted = {}
for key, val in bads_dict.items():
    bads_dict_sorted[key] = sorted(val)

# ensure the directory exists
fname_bads.parent.mkdir(parents=True, exist_ok=True)

with open(fname_bads, "w") as fout:
    json.dump(bads_dict_sorted, fout, indent=4, sort_keys=True)
    fout.write("\n")
