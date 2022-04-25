"""Mark bad temporal segments via MNE annotate functions and save."""

# %%
# Imports
import json
import sys
from pathlib import Path

from mne.preprocessing import (
    annotate_amplitude,
    annotate_break,
    annotate_muscle_zscore,
    annotate_nan,
)
from mne.utils import logger

from config import (
    FNAME_BADS_TEMPLATE,
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
fname_seg = Path(FNAME_SEGMENTS_TEMPLATE.format(sub=sub))
if fname_seg.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_seg))

# %%
# Get raw data
fpath_set = fpath_ds / "sourcedata" / "eeg_eeglab" / f"EMP{sub:02}.set"
raw = get_raw_data(fpath_set)

# %%
# Mark flat segments ... and potentially channels (if a channel is often flat)
annots_flat, bads_flat = annotate_amplitude(raw, flat=0)

# %%
# Mark block breaks as bad
annots_break = annotate_break(raw, t_start_after_previous=2, t_stop_before_next=2)

# %%
# Mark muscle artifacts
raw_copy = raw.copy()
raw_copy.load_data()
raw_copy.notch_filter([50, 100])

thresh_zscore = 4
annots_muscle, _ = annotate_muscle_zscore(
    raw_copy, threshold=thresh_zscore, ch_type="eeg", min_length_good=0.2
)

# %%
# Check for NaN
annots_nan = annotate_nan(raw)

# sanity check: there should be no NaNs ...
assert len(annots_nan) == 0

# %%
# combine annotations
annots_bad = annots_flat + annots_break + annots_muscle + annots_nan

# ensure we can set the bad annotations to raw
raw.set_annotations(raw.annotations + annots_bad)

# save
fname_seg.parent.mkdir(parents=True, exist_ok=True)
annots_bad.save(fname_seg, overwrite=overwrite)

# %%
# update bad channels, if we found previously unknown ones
fname_bads = Path(FNAME_BADS_TEMPLATE.format(sub=sub))
if len(bads_flat) > 0 and fname_bads.exists():
    with open(fname_bads, "r") as fin:
        bads_dict = json.load(fin)

    file_changed = False
    for bad in bads_flat:
        if bad not in bads_dict["bad_by_flat"]:
            bads_dict["bad_by_flat"].append(bad)
            file_changed = True

    if file_changed:
        bads_dict_sorted = {}
        for key, val in bads_dict.items():
            bads_dict_sorted[key] = sorted(val)

        with open(fname_bads, "w") as fout:
            json.dump(bads_dict_sorted, fout, indent=4, sort_keys=True)
            fout.write("\n")

elif not fname_bads.exists():
    logger.info("File on bad channels not found. Did you run 01_find_bads.py?")

# %%
