#!/usr/bin/env python

# some helper routines
import sys
import os
from time import time
import pickle

PKLDIR = "../pickle"

def is_interactive():
    """True if running script interactively."""
    return not hasattr(sys.modules['__main__'], '__file__')

def root_path():
    """Root of project for downloading stuff. The lemmatization requires
    a download of the wordnet lemma data - but its not too large."""
    return os.path.realpath(os.path.curdir) if is_interactive() else \
        os.path.dirname(__file__)

def timeit(func):
    """Simple decorator to print execution time."""
    def timed(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        print(f"{func.__name__} done in {time() - t0:0.2f} seconds")
        return res

    return timed

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
