"""Find subjects who did not perform well in the task.

Use signal detection theory to calculate sensitvity index d' (D prime),
and additionally calculate a binomial test of accuracy data against chance
level.

Based on these two indices, decide whether to exclude subjects from further
analysis.
"""
# %%
# Imports
import pandas as pd
from scipy import stats

from config import FPATH_DS, OVERWRITE_MSG

# %%
# Filepaths and settings
rejection_threshold = 0.6  # dprime threshold used to exclude participants
fpath_ds = FPATH_DS
overwrite = True

# %%
# Check overwrite
fname_bad_subjs = fpath_ds / "derivatives" / "bad_subjs.tsv"
if fname_bad_subjs.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_bad_subjs))


# %%
# define functions


def calculate_dPrime(behavior_dat):
    """Calculate the sensitvity index for a given subject.

    Does the following:
        - Calculates d'

    Parameters
    ----------
    behavior_dat : pd.DataFrame
        metadata of a given subject.

    Returns
    -------
    dprime : float
        the sensitvity index d'.
    """
    # hit rate
    hit_P = len(behavior_dat.query("behavior=='hit'")) / (
        len(behavior_dat.query("behavior=='hit'"))
        + len(behavior_dat.query("behavior=='miss'"))
    )
    # false alarm rate
    fa_P = len(behavior_dat.query("behavior=='falsealarm'")) / (
        len(behavior_dat.query("behavior=='falsealarm'"))
        + len(behavior_dat.query("behavior=='correctreject'"))
    )

    # z-scores
    hit_Z = stats.norm.ppf(hit_P)
    fa_Z = stats.norm.ppf(fa_P)

    # d-prime
    dPrime = hit_Z - fa_Z

    return dPrime


# %%
# Loads all subjs and check whether they should be excluded based on poor performance

subjs = []
dPrimes = []
p_vals = []
subj_exclude_dprime = []
subj_exclude_binom = []

for sub in range(1, 34):
    fpath_csv = fpath_ds / "sourcedata" / "events" / f"EMP{sub:02}_events.csv"
    df_beh = pd.read_csv(fpath_csv)
    # check if correct response was given
    df_beh = df_beh.assign(
        right_wrong=[
            1 if behav == "correctreject" or behav == "hit" else 0
            for behav in df_beh["behavior"]
        ]
    )

    dPrimes.append(calculate_dPrime(df_beh))

    p_vals.append(
        stats.binom_test(
            sum(df_beh["right_wrong"]),
            len(df_beh["right_wrong"]),
            p=0.5,
            alternative="greater",
        )
    )

    # check whether to exclude subj
    if dPrimes[-1] < rejection_threshold:
        # exclude: dprime is too low
        subj_exclude_dprime.append(1)
    else:
        subj_exclude_dprime.append(0)

    if p_vals[-1] > 0.001:
        # exclude: performance not significantly different from chance level
        subj_exclude_binom.append(1)
    else:
        subj_exclude_binom.append(0)

    subjs.append(sub)

# write subj data into df
data = {
    "subject_id": subjs,
    "dPrime": dPrimes,
    "pval_binom": p_vals,
    "exclude_based_on_dprime": subj_exclude_dprime,
    "exclude_based_on_binom": subj_exclude_binom,
}
ex_df = pd.DataFrame(data=data)
ex_df.to_csv(fname_bad_subjs, sep="\t", na_rep="n/a", index=False)
