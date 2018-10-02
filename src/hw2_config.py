#!/usr/bin/env python

import warnings
import logging
import seaborn as sns
import pprint
import nltk
import os

from hw2_utils import *

# General project constants
# NP: - I haven't been testing models with the entire dataset and I
#        wanted to make various things configurable
#     - I have been storing data in ./data/nips-papers.xlsx

# FIXME: where to store shared dataset?
# NP: the data I'm using here is the cleaned up version someone posted on piazza
# Here is the google drive link:
# 
# https://drive.google.com/file/d/180FBOXqxdyvzHihHsg_bAbYS-UF1WrKZ/view?usp=sharing
#
# It would be nice to be able to access shared dataset from our scripts, but
# I haven't looked into the python google-drive API (there is one though). Is
# there an alternative way for us to store our data?

DATADIR      = "../data"          # raw data directory
DATAFILE     = "nips-papers.xlsx" # name of dataset
PKLFILE      = "nips-papers.pkl"  # name of pickled data -- already cleaned
NLTKDIR      = "../nltk-data"     # NLTK downloads
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

# download nltk libraries in project root (they are in .gitignore)
nltk.data.path = [os.path.join(root_path(), NLTKDIR)]

# logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# pretty printing
pp = pprint.PrettyPrinter(indent=4)
