"""Script to prepare all saved files for submission to the EEGManyLabs project.

For details, see `organizer_documentation`.

In particular, we will create a "Scripts" and a "Data" directory.
In the "Scripts" directory, we will:

1. Copy this entire repository

In the "Data" directory, we will:

1. Create a README with information on excluded subjects and GIN.
2. Create a folder for each subject
3. Create sub-folders for each subject:
    a. Pre-processed time series data
    b. Removed ICA components
    c. Excluded trials
    d. Excluded sensors

The folders will be filled with the following contents:
- re: 3a
- re: 3b
- re: 3c
- re: 3d


"""
# %%
# Imports
from config import BAD_SUBJS, FPATH_DS

# %%
# Filepaths and settings
fpath_ds = FPATH_DS

handin_dir = fpath_ds.parent / "EMP_hand_in"

# %%
# Create directory for handing in files
handin_dir.mkdir(exist_ok=True)

# %%
# Create general README
readme = f"""All data is also stored on GIN, see:
https://gin.g-node.org/sappelhoff/eeg_manypipes_arc/

The following subjects were exluded from analysis: {list(BAD_SUBJS.keys())}.
For more information see the following files in the supplied scripts:

- `00_find_bad_subjs.py`
- `06b_check_autoreject.py`
- The `BAD_SUBJS` variable defined in `config.py`

"""
with open(handin_dir / "README.txt", "w") as fout:
    fout.write(readme)

# %%
