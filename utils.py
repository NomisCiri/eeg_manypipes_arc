"""General utility functions that are re-used in different scripts."""
import warnings
from pathlib import Path

import click
from pandas import read_csv
import mne
from mne.utils import logger
from scipy import stats


@click.command()
@click.option("--sub", type=int, help="Subject number")
@click.option("--fpath_ds", type=str, help="Data location")
@click.option("--overwrite", default=False, type=bool, help="Overwrite?")
def get_inputs(
    sub,
    fpath_ds,
    overwrite,
):
    """Parse inputs in case script is run from command line.

    See Also
    --------
    parse_overwrite
    """
    # strs to pathlib.Path
    fpath_ds = Path(fpath_ds) if fpath_ds else None

    # collect all in dict
    inputs = dict(
        sub=sub,
        fpath_ds=fpath_ds,
        overwrite=overwrite,
    )

    return inputs


def parse_overwrite(defaults):
    """Parse which variables to overwrite."""
    logger.info("\nParsing command line options...\n")
    inputs = get_inputs.main(standalone_mode=False, default_map=defaults)
    noverwrote = 0
    for key, val in defaults.items():
        if val != inputs[key]:
            logger.info(f"    > Overwriting '{key}': {val} -> {inputs[key]}")
            defaults[key] = inputs[key]
            noverwrote += 1
    if noverwrote > 0:
        logger.info(f"\nOverwrote {noverwrote} variables with command line options.\n")
    else:
        logger.info("Nothing to overwrite, use defaults defined in script.\n")

    return defaults


def get_raw_data(fpath_set):
    """Read data as MNE ``raw`` object, based on the path to a .set file."""
    # Read data, channel locations are automatically set
    # suppress a known warning that is not of consequence
    # https://github.com/mne-tools/mne-python/issues/10505
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="Not setting positions of 2 .*"
        )
    raw = mne.io.read_raw_eeglab(fpath_set, eog=["VEOG", "HEOG"])

    # Set some known metadata
    raw.info["line_freq"] = 50

    # Sanity check we have the expected number of events
    err_msg = f"    >>> {len(raw.annotations)} != the expected 1200"
    assert len(raw.annotations) == 1200, err_msg

    return raw

def get_behavioral_data(fpath_set):
    """ reads behavioral data for a given subject and returns it as a dataframe

    Args:
        fpath_set (string): path to behavioral data

    Returns:
        pd.dataFrame: dataframe containing metadata to be added to the epoched eeg data
    """
    metadata = read_csv(fpath_set)

    # Set some known metadata
    # Sanity check we have the expected number of events
    err_msg = f"    >>> {len(metadata)} != the expected 1200"
    assert len(metadata) == 1200, err_msg

    return metadata

def calculate_dPrime(behavior_dat):
    """calculates the sensitvity index for a given subject

    Args:
        behavior_dat (pd.dataFrame): a pandas dataframe containing a string that categorizes choices as hits, misses, correct rejections and false alarms

    Returns:
        dPrime(float): the senstivity index
    """
    
    hitP = (len(behavior_dat.query("behavior=='hit'"))/
            (len(behavior_dat.query("behavior=='hit'"))+len(behavior_dat.query("behavior=='miss'"))))
    # false alarm rate 
    faP  =  (len(behavior_dat.query("behavior=='falsealarm'"))/
             (len(behavior_dat.query("behavior=='falsealarm'"))+len(behavior_dat.query("behavior=='correctreject'"))))

    # z-scores
    hitZ = stats.norm.ppf(hitP)
    faZ  = stats.norm.ppf(faP)

    # d-prime
    dPrime = hitZ-faZ

    return dPrime