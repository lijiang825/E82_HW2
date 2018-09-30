#!/usr/bin/env python

import warnings
import seaborn as sns

# General project constants
# NP: - I haven't been testing models with the entire dataset and I
#        wanted to make it configurable to use stemming/lemmatization
#     - I also think it is worth having a random seed we use globally?

PRELIM      = True               # running preliminary analysis
PRELIM_FRAC = 0.2                # fraction of data to use in prelim. analysis
USE_LEMMA   = True               # use lemmatization
USE_STEM    = True               # stem words
RANDOM_SEED = 999                # seed for scikit-learn and random, eg. LDA
DATADIR     = "../data"          # raw data directory
DATA        = "nips-papers.xlsx" # name of dataset

warnings.filterwarnings("ignore")

# Any plotting configs
sns.set(style="white", color_codes=True) # minimal
