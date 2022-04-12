"""Highpass filter data and run ICA.

1. Load raw data
2. Load bad channels and attach to raw
3. Load bad segments and attach to raw
4. Highpass filter raw at 1Hz
5. Downsample raw to 128Hz
6. Run ICA on non-bad EEG channels (not EOG)
7. Save the ICA solution

"""

# %%
# Imports
import json
import sys
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import ICA
from mne.utils import logger

from config import (
    FNAME_BADS_TEMPLATE,
    FNAME_ICA_TEMPLATE,
    FNAME_SEGMENTS_TEMPLATE,
    FPATH_DS,
    OVERWRITE_MSG,
    PATH_NOT_FOUND_MSG,
    SUBJS,
)
from utils import get_raw_data, parse_overwrite

# %%
# Filepaths and settings

sub = 1
fpath_ds = FPATH_DS
overwrite = False

ica_rng = 1337

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
fname_ica = Path(FNAME_ICA_TEMPLATE.format(sub=sub))
if fname_ica.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_ica))

# %%
# Get raw data
fpath_set = fpath_ds / "sourcedata" / "eeg_eeglab" / f"EMP{sub:02}.set"
raw = get_raw_data(fpath_set)

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
# Downsample, to speed up ICA
raw = raw.resample(sfreq=128)

# %%
# Initialize an ICA object, using picard with settings equal to extended infomax
# Only estimate the first 20 components
assert len(raw.info["projs"]) == 0
ica = ICA(
    n_components=20,
    random_state=ica_rng,
    method="picard",
    fit_params=dict(ortho=False, extended=True),
)
# %%
# Get the channel indices of all channels that are *clean* and of type *eeg*
all_idxs = list(range(len(raw.ch_names)))

bad_idxs = [raw.ch_names.index(ii) for ii in raw.info["bads"]]
eog_idxs = mne.pick_types(raw.info, eog=True).tolist()

exclude_idxs = bad_idxs + eog_idxs
ica_idxs = list(set(all_idxs) - set(exclude_idxs))

# %%
# Fit our raw (high passed, downsampled) data with our ica object
# To speed up the computations, we only fit the middle 50% of the data
ica.fit(
    inst=raw,
    picks=ica_idxs,
    reject_by_annotation=True,
    start=np.percentile(raw.times, 25),
    stop=np.percentile(raw.times, 75),
)

# %%
# Save the ica object
ica.save(fname_ica, overwrite=overwrite)

# %%
