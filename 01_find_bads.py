"""Read data from source format (EEGLAB) and save as MNE format (.fif)."""

# %%
# Imports
import json
import sys
from pathlib import Path

import pyprep
from mne.utils import logger

from config import FNAME_BADS_TEMPLATE, FPATH_DS, OVERWRITE_MSG, SUBJS
from utils import get_raw_data, parse_overwrite

# %%
# Filepaths and settings

sub = 1
fpath_ds = FPATH_DS
overwrite = False

pyprep_rng = 42

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
    raise RuntimeError("The specified path to the data does not exist.")
if overwrite:
    logger.info("`overwrite` is set to ``True``.")

# %%
# Check overwrite
fname_bads = Path(FNAME_BADS_TEMPLATE.format(sub=sub))
if fname_bads.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_bads))

# %%
# Get raw data
fpath_set = fpath_ds / "sourcedata" / "eeg_eeglab" / f"EMP{sub:02}.set"
raw = get_raw_data(fpath_set)

# %%
# Find bad channels via pyprep
raw.load_data()
nc = pyprep.NoisyChannels(raw, random_state=pyprep_rng)
nc.find_all_bads()
bads_dict = nc.get_bads(as_dict=True)

# sanity check: POz was a reference channel, should be perfectly flat
assert "POz" in bads_dict["bad_by_flat"]

# %%
# Save the outputs under derivatives
bads_dict_sorted = {}
for key, val in bads_dict.items():
    bads_dict[key] = sorted(val)

# ensure the directory exists
fname_bads.parent.mkdir(parents=True, exist_ok=True)

with open(fname_bads, "w") as fout:
    json.dump(bads_dict, fout, indent=4, sort_keys=True)
    fout.write("\n")
