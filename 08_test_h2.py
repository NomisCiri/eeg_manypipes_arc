"""Test the hypotheses specified in the instructions.

Hypotheses read:
There are effects of image novelty (i.e., between images shown for the first time/new
vs. repeated/old images) within the time-range from 300–500 ms ...
a. ... on EEG voltage at fronto-central channels.
b. ... on theta power at fronto-central channels.
c. ... on alpha power at posterior channels.

"""

import itertools

# %%
# Imports
import os

import mne
import numpy as np
from mne.channels import find_ch_adjacency

from config import FPATH_DS, SUBJS, TRIGGER_CODES
from utils import catch  # , parse_overwrite

# %%
# Path and settings
fpath_ds = FPATH_DS
overwrite = True
# Roi & toi
ch_fronto_central = ["FCz", "FC1", "FC2", "FC3", "FC4", "Fz"]
ch_posterior = ["Poz", "Po3", "Po4", "Oz", "O1", "O2", "Po7" "Po8"]
toi_min = 0.3
toi_max = 0.5
# Settings for cluster
tfce = dict(start=0.4, step=0.4)

# List of all trigger combinations for a new image
triggers_new_list = list(
    itertools.product(
        list(TRIGGER_CODES[0].values()),
        ["new"],
        list(TRIGGER_CODES[2].values()),
        list(TRIGGER_CODES[3].values()),
    )
)
# List of all trigger combinations for an old image
triggers_old_list = list(
    itertools.product(
        list(TRIGGER_CODES[0].values()),
        ["old"],
        list(TRIGGER_CODES[2].values()),
        list(TRIGGER_CODES[3].values()),
    )
)

# %%
# Makes triggercodes for subsetting the epochs
triggers_new = [
    "/".join(map(str, triggers_new_list[i])) for i in range(0, len(triggers_new_list))
]

triggers_old = [
    "/".join(map(str, triggers_old_list[i])) for i in range(0, len(triggers_old_list))
]
# %%
# Reads in all epochs
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
# %%
#  Keep only existing subs
epochs_complete = list(filter(None.__ne__, epochs))
# %%
# Get a list of epochs in the desired timerange and with the desired channels.
# already put it into the format
# needed for permutation test (n_observations (subs), time, n_vertices (channels)).

# old images
epochs_old_list = list(
    [
        x[triggers_old]
        .crop(toi_min, toi_max)
        .pick_channels(ch_fronto_central)
        .average()
        .get_data()
        .transpose(1, 0)
        for x in epochs_complete
    ]
)
epochs_old_arr = np.vstack(epochs_old_list)
# new images
epochs_new_list = list(
    [
        x[triggers_new]
        .crop(toi_min, toi_max)
        .pick_channels(ch_fronto_central)
        .average()
        .get_data()
        .transpose(1, 0)
        for x in epochs_complete
    ]
)
epochs_new_arr = np.vstack(epochs_new_list)

# %%
# Calculate adjacency matrix between sensors from their locations
adjacency, _ = find_ch_adjacency(
    epochs_complete[1].pick_channels(ch_fronto_central).info, "eeg"
)

# Extract data: transpose because the cluster test requires channels to be last
# In this case, inference is done over items. In the same manner, we could
# also conduct the test over, e.g., subjects.

X = [epochs_old_arr, epochs_new_arr]

# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(
    X, tfce, adjacency=adjacency, n_permutations=100
)  # a more standard number would be 1000+
# %%
