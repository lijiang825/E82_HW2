#!/usr/bin/env python

# some helper routines
import sys
import os
from time import time

def is_interactive():
    """True if running script interactively."""
    return not hasattr(sys.modules['__main__'], '__file__')

def root_path():
    """Root of project for downloading stuff. The lemmatization requires
    a download of the wordnet lemma data - but its not too large."""
    return os.path.realpath(os.path.curdir) if is_interactive else \
        os.path.dirname(__file__)

def timeit(func):
    """Simple decorator to print execution time."""
    def timed(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        print(f"{func.__name__} done in {time() - t0:0.2f} seconds")
        return res

    return timed
