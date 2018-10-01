#!/usr/bin/env python

import warnings
import logging
import seaborn as sns
import pprint

# General project constants
# NP: - I haven't been testing models with the entire dataset and I
#        wanted to make various things configurable

DATADIR      = "../data"          # raw data directory
DATAFILE     = "nips-papers.xlsx" # name of dataset
SAMPLE       = True               # use only a sample of the data
SAMPLE_FRAC  = 0.2                # fraction of data to sample
EXTRACT_REFS = True               # Extract references from text section
REMOVE_REFS  = True               # Remove reference section from text
USE_LEMMA    = True               # use lemmatization
# USE_STEM     = True               # stem words
RANDOM_SEED  = 999                # seed for scikit-learn and random, eg. LDA

# General settings
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True) # minimal

# logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# pretty printing
pp = pprint.PrettyPrinter(indent=4)
