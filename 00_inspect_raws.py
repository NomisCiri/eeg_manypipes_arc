"""Inspect raw files manually."""
# %%
# imports
import mne

from config import FPATH_DS
from utils import get_raw_data

# %matplotlib #toggle interactive mode
# %%
# inspect raw data
sub = 4
fpath_set = FPATH_DS / "sourcedata" / "eeg_eeglab" / f"EMP{sub:02}.set"
raw = get_raw_data(fpath_set)
# raw.filter(l_freq=0.1, h_freq=40)
raw.plot(
    block=True,
    use_opengl=False,
    n_channels=len(raw.ch_names),
    bad_color="red",
    duration=20.0,
    clipping=None,
)

# %%
# inspect ica timecourse
fpath_ica = FPATH_DS / "derivatives" / f"EMP{sub:02}" / f"EMP{sub:02}_ica.fif.gz"
ica = mne.preprocessing.read_ica(fpath_ica)
# raw.load_data().filter(l_freq=0.1, h_freq=40)
ica.plot_sources(inst=raw)
