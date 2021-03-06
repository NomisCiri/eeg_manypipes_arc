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

# The raw data in .set format
FNAME_RAW_SET_TEMPLATE = os.path.join(
    str(FPATH_DS), "sourcedata", "eeg_eeglab", "EMP{sub:02}.set"
)
# The original events.csv files
FNAME_EVENTS_TEMPLATE = os.path.join(
    str(FPATH_DS), "sourcedata", "events", "EMP{sub:02}_events.csv"
)
# bad channels
FNAME_BADS_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_bad-channels.json"
)
# bad temporal segments
FNAME_SEGMENTS_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_bad-segments.txt"
)
# ICA solution
FNAME_ICA_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_ica.fif.gz"
)
# ICA components to exclude
FNAME_COMPONENTS_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_bad-components.json"
)
# raw data after ICA cleaning
FNAME_ICA_RAW_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_ica-raw.fif.gz"
)
# report for ICA
FNAME_REPORT_ICA_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_report-ica.html"
)
# epoched data
FNAME_EPOCHS_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_epo.fif.gz"
)
# epoched and CLEANED data
FNAME_EPO_CLEAN_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_clean-epo.fif.gz"
)
# Autoreject files
FNAME_AR_OBJECT_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_autoreject-obj.hdf5"
)
FNAME_AR_REJECT_LOG_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_reject-log.npz"
)
FNAME_AR_PLOT_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_reject-plot.pdf"
)
FNAME_BAD_EPOS_TEMPLATE = os.path.join(
    str(FPATH_DS), "derivatives", "EMP{sub:02}", "EMP{sub:02}_bad-epochs.json"
)
# Hypotheses tests
FNAME_HYPOTHESES_1_TEMPLATE = os.path.join(str(FPATH_DS), "derivatives", "h1", "{h}")
FNAME_HYPOTHESES_2_TEMPLATE = os.path.join(str(FPATH_DS), "derivatives", "h2", "{h}")
FNAME_HYPOTHESES_3_TEMPLATE = os.path.join(str(FPATH_DS), "derivatives", "h3", "{h}")
FNAME_HYPOTHESES_4_TEMPLATE = os.path.join(str(FPATH_DS), "derivatives", "h4", "{h}")

# H1 files
FNAME_REPORT_H1 = FPATH_DS / "derivatives" / "report-H1.html"
FNAME_REPORT_H2 = FPATH_DS / "derivatives" / "report-H2.html"
FNAME_REPORT_H3 = FPATH_DS / "derivatives" / "report-H3.html"
FNAME_REPORT_H4 = FPATH_DS / "derivatives" / "report-H4.html"

# Constants
# --------------------------------------------------------------------

BAD_SUBJS = {
    14: "bad performance, see 00_find_bad_subjs.py",
    31: "bad performance, see 00_find_bad_subjs.py",
    7: "odd artifacts in data, see 00_inspect_raws",
}
# originally, subjects from 1 to 33
SUBJS = np.array(list(set(range(1, 34)) - set(BAD_SUBJS)))

OVERWRITE_MSG = "\nfile exists and overwrite is False:\n\n>>> {}\n"
PATH_NOT_FOUND_MSG = "\npath not found\n\n>>> {}\n"
FPATH_DS_NOT_FOUND_MSG = (
    "Did not find the path:\n\n>>> {}\n"
    "\n>>Did you define the path to the data on your system in `config.py`? "
    "See the FPATH_DS variable!<<\n"
)

# Filtering
LOW_CUTOFF = 0.1
HIGH_CUTOFF = 100
HIGH_CUTOFF_TIMEDOMAIN = 40
NOTCH_FREQS = [50, 100]

# Events
TRIGGER_CODES = [
    {1: "man_made", 2: "natural"},
    {0: "new", 1: "old"},
    {1: "hit", 2: "miss", 3: "false_alarm", 4: "correct_rej", 9: "na"},
    {0: "sub_remembered", 1: "sub_forgotten", 9: "na"},
]

# This channel is flat
REF_CHANNEL = "POz"
