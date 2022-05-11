"""Script to prepare all saved files for submission to the EEGManyLabs project.

NOTE: Only run this script **after** having run the full pipeline (see README).

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
- re: 3a --> `EMP*_clean-epo.fif.gz`
- re: 3b --> `EMP*_ica.fif.gz`, `EMP*_report-ica.html`, `EMP*_bad-components.json`
- re: 3c --> `EMP*_bad-epochs.json`, `EMP*_bad-segments.txt`
- re: 3d --> `EMP*_reject-plot.pdf`, `EMP*_bad-channels.json`

NOTE: For copying files, we need to "unlock" via DataLad

"""
# %%
# Imports
import subprocess

from config import BAD_SUBJS, FPATH_DS, SUBJS

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
readme = f"""PLEASE READ THE FOLLOWING!

All data is also stored on GIN, see:
https://gin.g-node.org/sappelhoff/eeg_manypipes_arc/

Inspecting the derivatives stored on GIN may be very beneficial when
trying to understand the scripts.

The following subjects were exluded from analysis: {list(BAD_SUBJS.keys())}.
For more information see the following files in the supplied scripts:

- `00_find_bad_subjs.py`
- `06b_check_autoreject.py`
- The `BAD_SUBJS` variable defined in `config.py`

Description of subject-specific folders.

1. Pre-processed time series data
---------------------------------

The `EMP*_clean-epo.fif.gz` files contain the pre-processed epochs.

2. Removed ICA components
-------------------------

The `EMP*_ica.fif.gz` files are the MNE-Python ICA objects containing
the mixing and unmixing matrices.

The `EMP*_report-ica.html` files are report files to be opened inside
any Internet browser (e.g., Firefox). They contain information on
how many ICA components were obtained per subject, and which components
were rejected due to being identified as VEOG or HEOG artifact.

The `EMP*_bad-components.json` files contain a summary of all rejected
ICA components.

3. Excluded trials
------------------

The `EMP*_bad-epochs.json` files contain the finally rejected epochs.

The `EMP*_bad-segments.txt` files contain segment indices into the
raw data that were excluded from ICA decomposition, **before** epoching.
NOTE: These segments were treated as normal during pre-processing after
ICA. They were only excluded for ICA fitting.


4. Excluded sensors
-------------------

No sensors were excluded from the final pre-processed epochs.
We used the "autoreject" method, which either interpolates channels
locally (per epochs were they are considered bad), or rejects the
entire epoch if too many channels are bad locally.
The `EMP*_reject-plot.pdf` files provide an overview of interpolated
channels and rejected epochs (see also "3. Excluded trials")

The `EMP*_bad-channels.json` contains sensors that were excluded
from ICA decomposition, **before** epoching.
NOTE: These sensors were treated as normal during pre-processing
after ICA. They were only excluded for ICA fitting.
"""
with open(handin_dir / "README.txt", "w") as fout:
    fout.write(readme)

# %%
# Create subject folders
for sub in SUBJS[:1]:
    pass
