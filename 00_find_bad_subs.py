"""finds bad subjects and saves them as json.

based on either sensitvity index d'
or binomial test against chance.
"""
# %%
# Imports
import pandas as pd
from scipy import stats

from config import FPATH_DS

# %%
# Filepaths and settings
rejection_threshold = 0.6  # dprime threshold used to exclude participants
fpath_ds = FPATH_DS
overwrite = True
fpath_der = fpath_ds / "derivatives"

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
# loads all subjects and checks whether
# should be excluded based on poor performance

sub_ex = []
dPrime = []
p_val = []
sub_ex_d_bino = []

for sub in range(1, 34):
    fpath_set = fpath_ds / "sourcedata" / "events" / f"EMP{sub:02}_events.csv"
    behavior_dat = pd.read_csv(fpath_set)
    # check if correct response was given
    behavior_dat = behavior_dat.assign(
        right_wrong=[
            1 if behav == "correctreject" or behav == "hit" else 0
            for behav in behavior_dat["behavior"]
        ]
    )

    dPrime.append(calculate_dPrime(behavior_dat))

    p_val.append(
        stats.binom_test(
            sum(behavior_dat["right_wrong"]),
            len(behavior_dat["right_wrong"]),
            p=0.5,
            alternative="greater",
        )
    )

    # check based on which criterion we would exclude subs
    if (dPrime[-1] < rejection_threshold) & (p_val[-1] > 0.001):
        # rejected based on both, binomial test and dprime
        sub_ex_d_bino.append(3)
    elif (dPrime[-1] < rejection_threshold) & (p_val[-1] < 0.001):
        # rejected based on binomial test only
        sub_ex_d_bino.append(2)
    elif (dPrime[-1] > rejection_threshold) & (p_val[-1] > 0.001):
        # rejected based on dprime only
        sub_ex_d_bino.append(1)
    else:
        # not rejected
        sub_ex_d_bino.append(0)

    sub_ex.append(sub)
# write excluded ppts with their dprimes int df
ex_dat = {
    "subject_number": sub_ex,
    "dPrime": dPrime,
    "p_val": p_val,
    "ex_bi1_dp2_both3": sub_ex_d_bino,
}
ex_df = pd.DataFrame(data=ex_dat)
ex_df.to_csv(fpath_der / "bad_subs_dPrime.tsv", sep="\t", na_rep="n/a", index=False)
