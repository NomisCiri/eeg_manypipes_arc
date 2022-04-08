"""Stable configuration to be re-used across all scripts."""

import os
from pathlib import Path

import numpy as np

# Where to find the data (depending on user who runs the code)
# just add new users with additional "elif" logic clauses
#
# sourcedata and derivatives are always nested in the same way within the data
home = os.path.expanduser("~")
if "stefanappelhoff" in home:
    FPATH_DS = Path("/home/stefanappelhoff/Desktop/ds/eeg_manypipes_arc")
elif "ciranka" in home:
    FPATH_DS = Path("/... absolute path to data")
else:
    raise RuntimeError(f"unknown home: {home}. Add it to config.py!")


BAD_SUBJS = {
    99: "Add new Bad subjs like this (this is an example)",
}

# originally, subjects from 1 to 33
SUBJS = np.array(list(set(range(1, 34)) - set(BAD_SUBJS)))
