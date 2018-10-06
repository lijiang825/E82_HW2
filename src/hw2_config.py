#!/usr/bin/env python

import warnings
import logging
import seaborn as sns
import pprint
import nltk
import os
import sys
import os
from time import time
import pickle

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
PKLDIR       = "../pickle"        # store pickled objects
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

def is_interactive():
    """True if running script interactively."""
    return not hasattr(sys.modules['__main__'], '__file__')

def root_path():
    """Root of project for downloading stuff. The lemmatization requires
    a download of the wordnet lemma data - but its not too large."""
    return os.path.realpath(os.path.curdir) if is_interactive() else \
        os.path.dirname(__file__)

# download nltk libraries in project root (they are in .gitignore)
nltk.data.path = [os.path.join(root_path(), NLTKDIR)]

# logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# pretty printing
pp = pprint.PrettyPrinter(indent=4)

## -------------------------------------------------------------------
### General helper functions

def get_nltk_prereqs():
    """Download NLTK prereqs in root NLTKDIR."""
    nltk.download(['wordnet', 'punkt', 'stopwords', 'averaged_perceptron_tagger'],
                  download_dir=os.path.join(root_path(), NLTKDIR))

def timeit(func):
    """Simple decorator to print execution time."""
    def timed(*args, **kw):
        t0 = time()
        res = func(*args, **kw)
        t1 = time() - t0

        print(f"{func.__name__} finished in ", end='')
        if t1 > 60:
            print(f"{t1 / 60.:0.2f} minutes")
        else:
            print(f"{t1:0.2f} seconds")
        return res

    return timed

def get_save_name(sections):
    """Return name of saved data composed of sections."""
    return '_'.join(sections) if isinstance(sections, list) else sections

def get_save_path(name):
    """Return filepath and whether it exists in the in pickle directory."""
    pklpath = os.path.join(root_path(), PKLDIR, name)
    return pklpath, os.path.exists(pklpath)

def pickle_load(filename="models.pkl"):
    """Load pickled objects, eg. models, datasets."""
    pklpath = os.path.join(root_path(), PKLDIR, filename)
    if os.path.exists(pklpath):
        with open(pklpath, "rb") as f:
            return pickle.load(f)
    return dict()

def pickle_save(obj, filename="models.pkl"):
    """OBJ should be a dictionary with all of the objects to pickle."""
    pklpath = os.path.join(root_path(), PKLDIR, filename)
    with open(pklpath, "wb") as f:
    	pickle.dump(obj, f)

def load_tokenized_data(filename='nips-tokenized.pkl'):
    """Load preprocessed tokenized/lemmatized data."""
    dpath = os.path.join(root_path(), DATADIR, 'nips-tokenized.pkl')
    return pickle_load(dpath) if os.path.exists(dpath) else None
