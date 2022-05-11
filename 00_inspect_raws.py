"""Inspect raw data interactively."""
# %%
# Imports
import mne

from config import (
    FNAME_ICA_TEMPLATE,
    FNAME_RAW_SET_TEMPLATE,
    FPATH_DS,
    FPATH_DS_NOT_FOUND_MSG,
)
from utils import get_raw_data

# %%
# Filepaths and settings
sub = 4  # change interactively

if not FPATH_DS.exists():
    raise RuntimeError(FPATH_DS_NOT_FOUND_MSG.format(FPATH_DS))

# %%
# Load raw data
raw = get_raw_data(FNAME_RAW_SET_TEMPLATE.format(sub=sub))
raw.load_data()
raw.filter(l_freq=0.1, h_freq=40)

# %%
# Inspect raw data
raw.plot(
    block=True,
    use_opengl=False,
    n_channels=len(raw.ch_names),
    bad_color="red",
    duration=20.0,
    clipping=None,
)

# %%
# Inspect ica timecourse
fname_ica = FNAME_ICA_TEMPLATE.format(sub=sub)
if fname_ica.exists():
    ica = mne.preprocessing.read_ica(fname_ica)
    ica.plot_sources(inst=raw)
else:
    print("ICA data does not (yet) exist.")

# %%
