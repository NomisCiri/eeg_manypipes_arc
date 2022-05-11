"""Inspect raw data interactively."""
# %%
# Imports
import mne

from config import FPATH_DS
from utils import get_raw_data

# %%
# Filepaths and settings
sub = 4  # change interactively

fpath_set = FPATH_DS / "sourcedata" / "eeg_eeglab" / f"EMP{sub:02}.set"
fpath_ica = FPATH_DS / "derivatives" / f"EMP{sub:02}" / f"EMP{sub:02}_ica.fif.gz"

# %%
# Load raw data
raw = get_raw_data(fpath_set)
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
if fpath_ica.exists():
    ica = mne.preprocessing.read_ica(fpath_ica)
    ica.plot_sources(inst=raw)
else:
    print("ICA data does not (yet) exist.")

# %%
