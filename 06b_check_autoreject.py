"""Inspect autoreject outputs."""

# %%
# Imports
import json
from pathlib import Path

import autoreject
import pandas as pd

from config import FNAME_AR_OBJECT_TEMPLATE, FNAME_BAD_EPOS_TEMPLATE, SUBJS

# %%
# Filepaths and settings

columns = ["subject", "n_rejected_epochs", "consensus", "n_interpolate"]
data = {i: [] for i in columns}
for sub in SUBJS:

    # load autoreject estimates
    fname_ar = Path(FNAME_AR_OBJECT_TEMPLATE.format(sub=sub))
    ar = autoreject.read_auto_reject(fname_ar)

    # load rejected epochs
    fname_bad_epos = Path(FNAME_BAD_EPOS_TEMPLATE.format(sub=sub))
    with open(fname_bad_epos, "r") as fin:
        bad_epos_dict = json.load(fin)

    n_rejected = len(bad_epos_dict["bad_epochs_idxs_0-based"])

    # report
    data["subject"] += [sub]
    data["n_rejected_epochs"] += [n_rejected]
    data["consensus"] += [ar.consensus_["eeg"]]
    data["n_interpolate"] += [ar.n_interpolate_["eeg"]]

df = pd.DataFrame.from_dict(data)
df
# %%
