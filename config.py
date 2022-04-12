"""Stable configuration to be re-used across all scripts."""

import os
from pathlib import Path

import numpy as np

# File paths
# --------------------------------------------------------------------

# Where to find the data (depending on user who runs the code)
# just add new users with additional "elif" logic clauses
#
# sourcedata and derivatives are always nested in the same way within the data
home = os.path.expanduser("~")
if "stefanappelhoff" in home:
    FPATH_DS = Path("/home/stefanappelhoff/Desktop/ds/eeg_manypipes_arc")
elif "appelhoff" in home:
    FPATH_DS = Path("/mnt/home/appelhoff/emp/ds/eeg_manypipes_arc")
elif "ciranka" in home:
    FPATH_DS = Path("/mnt/home/ciranka/emp/ds/eeg_manypipes_arc")
elif "runner" in home:
    # GitHub Actions CI
    FPATH_DS = (
        Path(home)
        / "work"
        / "eeg_manypipes_arc"
        / "eeg_manypipes_arc"
        / "eeg_manypipes_arc"
    )
    if os.name == "nt":
        FPATH_DS = Path(r"D:\a\eeg_manypipes_arc\eeg_manypipes_arc\eeg_manypipes_arc")

else:
    raise RuntimeError(f"unknown home: {home}. Add it to config.py!")


FNAME_BADS_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_bad-channels.json"
)
FNAME_SEGMENTS_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_bad-segments.txt"
)

# Constants
# --------------------------------------------------------------------

BAD_SUBJS = {
    14: "bad performance, see 00_find_bad_subs.py",
    31: "bad performance, see 00_find_bad_subs.py",
}
# originally, subjects from 1 to 33
SUBJS = np.array(list(set(range(1, 34)) - set(BAD_SUBJS)))

OVERWRITE_MSG = "\nfile exists and overwrite is False:\n\n>>> {}\n"
PATH_NOT_FOUND_MSG = "\npath not found\n\n>>> {}\n"
