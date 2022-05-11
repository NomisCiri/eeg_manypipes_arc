"""Script to prepare all saved files for submission to the EEGManyLabs project.

For details, see `organizer_documentation`.

In particular, we will create a "Scripts" and a "Data" directory.
In the "Scripts" directory, we will:

1. Copy this entire repository (via git clone from GitHub)

In the "Data" directory, we will:

1. Create a README with information on excluded subjects and GIN.
2. Create a folder for each subject
3. Create sub-folders for each subject:
    a. Pre-processed time series data
    b. Removed ICA components
    c. Excluded trials
    d. Excluded sensors

The folders will be filled with the following contents:
- re: 3a --> The pre-processed epochs: `EMP*_clean-epo.fif.gz`
- re: 3b -->
- re: 3c -->
- re: 3d -->


"""
# %%
# Imports
import subprocess

from config import BAD_SUBJS, FPATH_DS

# %%
# Filepaths and settings
fpath_ds = FPATH_DS

handin_dir = fpath_ds.parent / "EMP_hand_in"
scripts_dir = handin_dir / "Scripts"
data_dir = handin_dir / "Data"


# %%
# Create directories for handing in files
handin_dir.mkdir(exist_ok=True)
scripts_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)

# %%
# Copy entire repo to scripts, using git
repo = "https://github.com/NomisCiri/eeg_manypipes_arc.git"
dest = str(scripts_dir)

try:
    subprocess.run(
        ["git", "clone", repo, dest], check=True, stdout=subprocess.PIPE
    ).stdout
except subprocess.CalledProcessError:
    print(f"Error when cloning from {repo}. Assuming the folder already exists.")

# %%
# Create general README
readme = f"""All data is also stored on GIN, see:
https://gin.g-node.org/sappelhoff/eeg_manypipes_arc/

Inspecting the derivatives stored on GIN may be very beneficial when
trying to understand the scripts.

The following subjects were exluded from analysis: {list(BAD_SUBJS.keys())}.
For more information see the following files in the supplied scripts:

- `00_find_bad_subjs.py`
- `06b_check_autoreject.py`
- The `BAD_SUBJS` variable defined in `config.py`

"""
with open(handin_dir / "README.txt", "w") as fout:
    fout.write(readme)

# %%
