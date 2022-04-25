[![Run analysis](https://github.com/NomisCiri/eeg_manypipes_arc/actions/workflows/run_analysis.yml/badge.svg?branch=main&event=push)](https://github.com/NomisCiri/eeg_manypipes_arc/actions/workflows/run_analysis.yml)

# eeg_manypipes_arc

This project contains all code to reproduce the analyses of the EEG manypipes project

`sourcedata` and `derivatives` are stored on GIN:
[https://gin.g-node.org/sappelhoff/eeg_manypipes_arc](https://gin.g-node.org/sappelhoff/eeg_manypipes_arc)

## Installation

To run the code, you need to install the required dependencies first.
We recommend that you follow these steps
(assumed to be run from the root of this repository):

1. Download Miniconda for your system: https://docs.conda.io/en/latest/miniconda.html
   (this will provide you with the `conda` command)
1. Use `conda` to install `mamba`: `conda install mamba -n base -c conda-forge`
   (for more information, see: https://github.com/mamba-org/mamba;
   NOTE: We recommend that you install `mamba` in your `base` environment.)
1. Use the `environment.yml` file in this repository to create the `emp` ("EEGManyPipelines") environment:
   `mamba env create -f environment.yml`
1. Activate the environment as usual with `conda activate emp`
1. After the **first** activation, run the following to activate pre-commit hooks: `pre-commit install`

## Obtaining the data

We recommend that you make use of the data hosted on GIN via
[Datalad](https://handbook.datalad.org/en/latest/index.html).

If you followed the installation steps above, you should **almost** have a working installation of
Datalad in your environment.
The last step that is (probably) missing, is to install
[git-annex](https://git-annex.branchable.com/).

Depending on your operating system, do it as follows:
    - ubuntu: `mamba install -c conda-forge git-annex`
    - macos: `brew install git-annex` (use [Homebrew](https://brew.sh/))
    - windows: `choco install git-annex` (use [Chocolatey](https://chocolatey.org/))

Use the following steps to download the data:

1. clone: `datalad clone https://gin.g-node.org/sappelhoff/eeg_manypipes_arc`
1. go to root of cloned dataset: `cd eeg_manypipes_arc`
1. get a specific piece of the data `datalad get sourcedata/eeg_eeglab/EMP01.set`
1. ... or get all data: `datalad get *`

Note that if you do not `get` all the data (step 4. above), the data that you did not `get`
is not actually present on your system.
There is merely a symbolic link to a remote location (GIN).
Furthermore, the entire EEG data (even after `get`) is "read only";
if you need to edit the files (not recommended), you can run `datalad unlock *`.

## Screen the raw data

To screen some subject's raw data, follow these steps:

1. Activate the `emp` environment
1. Start `ipython` from the command line (from the root of this repository)
1. Paste the following code snippets, adjusting the `sub` variable to your liking
1. (optional) Uncomment and adjust the "filter" comment to filter the data

Note that this assumes you have already downloaded the data and added the path
to your downloaded data to `config.py`.

```python
import mne
from config import FPATH_DS
from utils import get_raw_data
sub = 1
fpath_set = FPATH_DS / "sourcedata" / "eeg_eeglab" / f"EMP{sub:02}.set"
raw = get_raw_data(fpath_set)
#raw.filter(l_freq=0.1, h_freq=40)
raw.plot(
    block=True,
    use_opengl=False,
    n_channels=len(raw.ch_names),
    bad_color="red",
    duration=20.0,
    clipping=None,
)
```

## Continuous integration

Under `.github/workflows/run_analysis.yml` we have specified a test workflow that may be
helpful for you to inspect.


## Run several subjs after one another

Use something like this from a command line prompt:

```shell
for i in {1..33}
do
    python -u 01_find_bads.py \
        --sub=$i \
        --overwrite=True
done
```
