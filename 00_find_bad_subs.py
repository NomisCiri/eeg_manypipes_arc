"""finds bad subjects and saves them as json.

based on either sensitvity index d'
or binomial test against chance.
the latter works bc of a clever task design
To reproduce, you have to delete the bad subs from config.
"""
# %%
# Imports
import pandas as pd
from scipy.stats import binom_test

from config import FPATH_DS, SUBJS

from utils import calculate_dPrime, get_behavioral_data

# %%
# Filepaths and settings
rejection_threshold = 0.6  # dprime threshold used to exclude participants
fpath_ds = FPATH_DS
overwrite = False
fpath_der = fpath_ds / "derivatives"

# %%

subex = []
dPrimeEx = []

subexBino = []
binoEx = []
# based on d'
for sub in SUBJS:
    fpath_set = fpath_ds / "sourcedata" / "events" / f"EMP{sub:02}_events.csv"
    behavior_dat = get_behavioral_data(fpath_set)
    behavior_dat = behavior_dat.assign(
        right_wrong=[
            1 if behav == "correctreject" or behav == "hit" else 0
            for behav in behavior_dat["behavior"]
        ]
    )

    dPrime = calculate_dPrime(behavior_dat)  # get dprime
    p_val = binom_test(
        sum(behavior_dat["right_wrong"]),
        len(behavior_dat["right_wrong"]),
        p=0.5,
        alternative="greater",
    )  # get binmoial test

    # collect subjects to exclude based on d'
    if dPrime < rejectionThreshold:
        subex.append(sub)
        dPrimeEx.append(dPrime)

    # collect subjects to exclude based on binomial test against chance
    if p_val > 0.001:
        subexBino.append(sub)
        binoEx.append(p_val)
# write excluded ppts with their dprimes int df
exDat = {"subjectNumber": subex, "dPrime": dPrimeEx}
exdf = pd.DataFrame(data=exDat)
exdf.to_json(fpath_der / "bad_subs_dPrime.json")

exDatBino = {"subjectNumber": subexBino, "p_val": binoEx}
exdfBino = pd.DataFrame(data=exDatBino)
exdfBino.to_json(fpath_der / "bad_subs_binomial.json")

# %%
