[![Run analysis](https://github.com/NomisCiri/eeg_manypipes_arc/actions/workflows/run_analysis.yml/badge.svg)](https://github.com/NomisCiri/eeg_manypipes_arc/actions/workflows/run_analysis.yml)

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

If you followed the installation steps above, you should have a working installation of
Datalad in your environment already.
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

## Continuous integration

Under `.github/workflows/run_analysis.yml` we have specified a test workflow that may be
helpful for you to inspect.
