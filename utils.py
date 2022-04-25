"""General utility functions that are re-used in different scripts."""
import warnings
from pathlib import Path

import click
import mne
from mne.utils import logger

from config import TRIGGER_CODES


@click.command()
@click.option("--sub", type=int, help="Subject number")
@click.option("--fpath_ds", type=str, help="Data location")
@click.option("--overwrite", default=False, type=bool, help="Overwrite?")
@click.option("--interactive", default=False, type=bool, help="Interactive?")
def get_inputs(
    sub,
    fpath_ds,
    overwrite,
    interactive,
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
        interactive=interactive,
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

    # IO1 and IO2 are also EOG channels: very close to the eyes
    raw = raw.set_channel_types({"IO1": "eog", "IO2": "eog"})

    # Sanity check we have the expected number of events
    err_msg = f"    >>> {len(raw.annotations)} != the expected 1200"
    assert len(raw.annotations) == 1200, err_msg

    return raw


def event2id(event_str):
    """Convert a 4-digit string (an event) to a human readable description."""
    event_id = []
    for event_factor, trigger_code in zip(event_str, TRIGGER_CODES):
        event_id.append(trigger_code[int(event_factor)])

    return "/".join(event_id)
