"""Test the first hypothesis.

> There is an effect of scene category
> (i.e., a difference between images showing man-made vs. natural environments)
> on the amplitude of the N1 component,
> i.e. the first major negative EEG voltage deflection.

"""

# %%
# Imports

import mne
from tqdm.auto import tqdm

from config import FNAME_EPOCHS_TEMPLATE, SUBJS

# %%
# Filepaths and settings
sub = 1

# epochs as loaded are from -1.5 to 2.5, but this is too long for this ERP analysis
crop = (-0.2, 0.7)

# epochs as loaded are not yet baseline corrected, do it now
baseline = (None, 0)

keys = ["natural", "man_made"]

# %%
# Read all epochs and make ERPs per subj

evokeds = {i: [] for i in keys}
for sub in tqdm(SUBJS):
    fname = FNAME_EPOCHS_TEMPLATE.format(sub=sub)
    epochs = mne.read_epochs(fname, preload=True)
    epochs.crop(*crop)
    epochs.apply_baseline(baseline=baseline)

    for key in keys:
        evokeds[key] += [epochs[key].average()]


# %%
