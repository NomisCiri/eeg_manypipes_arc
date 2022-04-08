"""General utility functions that are re-used in different scripts."""

from pathlib import Path

import click
from mne.utils import logger


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
