"""Filter data, make epochs.

1. Load ICA cleaned data
1. Filter data (lowpass, highpass)
1. Make epochs
1. Save

"""

# %%
# Imports
import sys
from pathlib import Path

import mne
import pandas as pd
from mne.utils import logger

from config import (
    FNAME_EPOCHS_TEMPLATE,
    FNAME_EVENTS_TEMPLATE,
    FNAME_ICA_RAW_TEMPLATE,
    FPATH_DS,
    FPATH_DS_NOT_FOUND_MSG,
    HIGH_CUTOFF,
    LOW_CUTOFF,
    NOTCH_FREQS,
    OVERWRITE_MSG,
    SUBJS,
)
from utils import event2id, parse_overwrite

# %%
# Filepaths and settings

sub = 1
overwrite = True

t_min_max_epochs = (-1.5, 2.5)


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
fname_epochs = Path(FNAME_EPOCHS_TEMPLATE.format(sub=sub))
if fname_epochs.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_epochs))

# %%
# Get raw data
fpath_fif = Path(FNAME_ICA_RAW_TEMPLATE.format(sub=sub))
raw = mne.io.read_raw_fif(fpath_fif)
raw.load_data()

# %%
# Filter (highpass, lowpass, notch)
raw = raw.filter(l_freq=LOW_CUTOFF, h_freq=None)
raw = raw.filter(l_freq=None, h_freq=HIGH_CUTOFF)
raw = raw.notch_filter(freqs=NOTCH_FREQS)

# %%
# Make epochs


def str2int(str_input):
    """Convert a str to int."""
    return int(str_input)


# Prepare event_id for Epochs
events, event_id = mne.events_from_annotations(raw, event_id=str2int)

this_event_id = {}
for key, value in event_id.items():
    this_event_id[event2id(key)] = value

# prepare metadata
fname_events = Path(FNAME_EVENTS_TEMPLATE.format(sub=sub))
metadata = pd.read_csv(fname_events, sep=",")
assert metadata.shape[0] == 1200

# form epochs (only save EEG channels)
tmin, tmax = t_min_max_epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id=this_event_id,
    tmin=tmin,
    tmax=tmax,
    metadata=metadata,
    baseline=None,
    picks=["eeg"],
    preload=True,
)
assert len(epochs) == 1200

# %%
# Save
epochs.save(fname_epochs, overwrite=overwrite)

# %%
