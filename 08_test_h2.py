"""Test the hypotheses specified in the instructions.

Hypotheses read:
There are effects of image novelty (i.e., between images shown for the first time/new
vs. repeated/old images) within the time-range from 300–500 ms ...
a. ... on EEG voltage at fronto-central channels.
b. ... on theta power at fronto-central channels.
c. ... on alpha power at posterior channels.

"""

# %%
# Imports
import os

import mne

from config import FPATH_DS, SUBJS, TRIGGER_CODES
from utils import catch  # , parse_overwrite

# %%
# Path and settings
fpath_ds = FPATH_DS
overwrite = True
ch_fronto_central = ["FCz", "FC1", "FC2", "FC3", "FC4", "Fz"]
ch_posterior = ["Poz", "Po3", "Po4", "Oz", "O1", "O2", "Po7" "Po8"]
tmin = 300
tmax = 500
triggers_new = [
    list(TRIGGER_CODES[0].keys()),
    [0],
    list(TRIGGER_CODES[2].keys()),
    list(TRIGGER_CODES[3].keys()),
]
# %%
# reads in all epochs
epochs = [
    catch(
        lambda: mne.read_epochs(
            fname=os.path.join(
                str(fpath_ds),
                "derivatives",
                f"EMP{sub:02}",
                f"EMP{sub:02}_clean-epo.fif.gz",
            )
        )
    )
    for sub in SUBJS
]