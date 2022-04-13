"""finds bad subjects based on the senstivity index dprime in a signal detection task and saves them as json"""
# %%
# Imports
import json
import sys
import pandas as pd
from pathlib import Path
from scipy import stats

from config import (
    FNAME_BADS_TEMPLATE,
    FPATH_DS,
    OVERWRITE_MSG,
    PATH_NOT_FOUND_MSG,
    SUBJS,
)

#%% settings
from utils import get_behavioral_data, calculate_dPrime

# %%
# Filepaths and settings
rejectionThreshold=0.8# dprime after which we exclude participants
fpath_ds = FPATH_DS
overwrite = False
fpath_der=fpath_ds / "derivatives"

# %%

subex=[]
dPrimeEx=[]

for sub in SUBJS:
    fpath_set = fpath_ds / "sourcedata" / "events" / f"EMP{sub:02}_events.csv"
    behavior_dat = get_behavioral_data(fpath_set)
    dPrime=calculate_dPrime(behavior_dat)
    
    #collect subjects to exclude
    if dPrime<rejectionThreshold:
        subex.append(sub)
        dPrimeEx.append(dPrime)
#write excluded ppts with their dprimes int df
exDat={"subjectNumber":subex, "dPrime":dPrimeEx}    
exdf = pd.DataFrame(data=exDat)
exdf.to_json(fpath_der / "bad_subs.json")


# %%
