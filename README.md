[![Run analysis](https://github.com/NomisCiri/eeg_manypipes_arc/actions/workflows/run_analysis.yml/badge.svg?branch=main&event=push)](https://github.com/NomisCiri/eeg_manypipes_arc/actions/workflows/run_analysis.yml)
[![DOI](https://zenodo.org/badge/476738961.svg)](https://zenodo.org/badge/latestdoi/476738961)

# eeg_manypipes_arc

This project contains all code to reproduce the analyses of the
[EEG manypipes project](https://www.eegmanypipelines.org/).

It is hosted on GitHub: https://github.com/NomisCiri/eeg_manypipes_arc

This is a contribution by
[Stefan Appelhoff](https://stefanappelhoff.com/),
[Simon Ciranka](https://orcid.org/0000-0002-2067-9781),
and Casper Kerrén from
the Center of Adaptive Rationality
([ARC](https://www.mpib-berlin.mpg.de/research/research-centers/adaptive-rationality))

Original documentation provided by the organizers can be found in the `organizer_documentation` directory.

`sourcedata` and `derivatives` of this project are stored on GIN:
[https://gin.g-node.org/sappelhoff/eeg_manypipes_arc](https://gin.g-node.org/sappelhoff/eeg_manypipes_arc)

The report for the analysis is in `REPORT.md`

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
if you need to edit or overwrite the files (not recommended), you can run `datalad unlock *`.

## Continuous integration

Under `.github/workflows/run_analysis.yml` we have specified a test workflow that may be
helpful for you to inspect.

## Description of files

- files unrelated to analysis
    - `LICENSE`, detailing how our work is licensed
    - `README.md`, the information that you currently read
    - `setup.cfg`, a file to configure different software tools to work well with each other (black, flake8, ...)
    - `CITATION.cff`, metadata on how to cite this code
    - `.gitignore`, which files not to track in the version control system
    - `environment.yml`, needed to install software dependencies (see also "Installation" above)
    - `.pre-commit-config.yaml`, configuration for ["pre-commit hooks"](https://pre-commit.com/) that ease software development
    - `organizer_documentation/*.pdf`, the original documentation provided by the EEG Many Pipelines project organizers
    - `.github/workflows/run_analysis.yml`, a continuous integration workflow definition for [GitHub Actions](https://github.com/features/actions)

All other files are related to the analysis.

- `REPORT.md`, containing four short paragraphs on the analysis of the four hypotheses
- `config.py`, definitions of stable **variables** that are reused throughout other scripts; for example file paths
- `utils.py`, definitions of **functions** that are reused throughout other scripts

The Python script that are doing the heavy lifting have names that are prefixed with
two integers `00`, `01`, `02`, ...
This indicates the order in which to run the scripts.

The `00` are optional to run.

- `00_find_bad_subjs.py`, to find subjects to exclude from analysis based on behavioral performance (see `BAD_SUBJS` variable in `config.py`)
- `00_inspect_raws.py`, to interactively inspect raw EEG data
- `00_prepare_handin.py`, only to prepare all files for handing in for the EEGManyPipelines submission

The preprocessing scripts are those from `01` to `06`.
These operate on single subjects.

- `01_find_bads.py`, finding bad channels using [pyprep](https://github.com/sappelhoff/pyprep)
- `02_mark_bad_segments.py`, marking bad temporal segments using MNE-Python automatic methods
- `03_run_ica.py`, running ICA, excluding previously found bad channels and segments
- `04_inspect_ica.py`, find and exclude bad ICA components
- `05_make_epochs.py`, epoch the data
- `06_run_autoreject.py`, interpolate channels
- `06b_check_autoreject.py`, provide a summary of interpolated channels

Note that these scripts can be easily run from the command line and that you can specify
certain arguments there (see the scripts for more detail).
This allows running several subjects from the command line like below:

```shell
for i in {1..33}
do
    python -u 01_find_bads.py \
        --sub=$i \
        --overwrite=True \
        --fpath_ds="path/to/my/dataset"
done
```

Finally, there is one script for testing each of the four hypotheses.

- `07_test_h1.py`, for hypothesis 1
- `08_test_h2.py`, for hypothesis 2
- `09_test_h3.py`, for hypothesis 3
- `10_test_h4.py`, for hypothesis 4

---

**All outputs of these analyses are stored on GIN**

[https://gin.g-node.org/sappelhoff/eeg_manypipes_arc](https://gin.g-node.org/sappelhoff/eeg_manypipes_arc)
