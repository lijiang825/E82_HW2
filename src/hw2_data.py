#!/usr/bin/env python

"""
This script has functions to cleanup the data a bit more.

Functions: 
 - creates lists of references from the text of the articles (from the 
   References section), adding a new column "refs" to the data.
 - remove the Reference section from the text

Both of these cleaning steps are on by default, but can be disabled in
the get_data function.
"""

# FIXME: where to store shared dataset?
# NP: the data I'm using here is the cleaned up version someone posted on piazza
# Here is the google drive link:
# 
# https://drive.google.com/file/d/180FBOXqxdyvzHihHsg_bAbYS-UF1WrKZ/view?usp=sharing
#
# It would be nice to be able to access shared dataset from our scripts, but
# I haven't looked into the python google-drive API (there is one though). Is
# there an alternative way for us to store our data?
# 
# NP: I have been storing data in ./data/nips-papers.xlsx
# It is of the form:
# Index(['id', 'year', 'title', 'abstract', 'text'], dtype='object')

import pandas as pd
import numpy as np
import os

from hw2_utils import *         # helper functions
from hw2_config import *        # project constants

## -------------------------------------------------------------------
### Pull out references

def get_references(data):
    """
    Pulls out references from papers, returns a list of references indexed by
    paper ID. Only returns references from papers where a reference section was
    actually found. 

    Notes: 
      - some papers are truncated / missing references section. 
      - Also the references appear to be truncated in cases (eg. 6556).
      - the text is split by a reference regex that may match in multiple places,
        so the last element is the correct one. However, the entire list is returned
        here to enable later processing if desired (eg. removing the reference 
        section).

    The formats that are parsed are:
    References: (sometimes capitalized, eg. 1)
    [1] or (1) or 1. ...
    [2] or (2) or 2. ...
    ...

    However, some are just formatted as 
    References:
    ...
    And these are ignored for now since they are hard to parse.

    There seem to be ~2500 texts w/o detected reference sections in the cleaned 
    version of the data posted on piazza.
    """
    refs = data.text.str.split(
        r"\s+[Rr][Ee][Ff][Ee][Rr][Ee][Nn][Cc][Ee][Ss]\s+(?=[\[\(][0-9]{1,3}|[0-9]{1,3}[.]\s+)")
    mask = (refs.apply(len) > 1) # ignore texts with no detected references
    refs = refs[data.text[mask].index]
    return refs

def split_references(refs):
    """
    Split reference text into references where possible.

    They could be of forms: [1] ..., (1) ..., or 1. ...
    Others are ignored (hopefully).

    There are also assumed to only be upto 3 digits in a reference to avoid possible
    complication with years.
    """
    return refs.str.split(
        r"(?:^|\A|\n)\s*(?:[\[\(][0-9]{1,3}[\]\)]|[0-9]{1,3}[.]\s+)\s*")\
                   .apply(lambda x: [i for i in x if len(i) > 0])
    
def add_reference_column(data, **kwargs):
    """Add references to data. Pass pre-computed 'refs' to avoid recomputation."""
    refs = kwargs["refs"] if "refs" in kwargs else get_references(data)

    # Only use the last section
    data["refs"] = split_references(refs.apply(lambda x: (x[-1:])[0]))
    return data

def remove_reference_section(data, **kwargs):
    """
    Remove the reference section from text. The text is split into possible 
    reference sections by `get_references`, and the last section, presumably the
    actual reference section is removed.

    The split sections, minus the last one, are joined back together with 
    ' references '.
    """
    refs = kwargs["refs"] if "refs" in kwargs else get_references(data)
    # a couple have random refs, but the last section should be right
    data.text[refs.index] = refs.apply(lambda x: ' references '.join(x[:-1]))
    return data

def get_data(root=None, data=DATA, datadir=DATADIR, add_refs=True, remove_refs=True,
             frac=PRELIM_FRAC):
    """
    Loads data. Data location can be specified here or defined as constants
    in hw2_config.

    Options:
      - add column with list of references extracted from text
      - remove the references section from article text
    """
    rootdir = root_path() if root is None else root
    datadir = os.path.join(rootdir, DATADIR)
    datapath = os.path.join(datadir, DATA)
    dat = pd.read_excel(datapath)
    refs = None
    
    # Use only a fraction of data for preliminary runs
    if PRELIM and PRELIM_FRAC:
        dat = dat.sample(frac=PRELIM_FRAC, random_state=RANDOM_SEED)

    # add references to data
    if add_refs:
        refs = get_references(dat)
        dat = add_reference_column(dat, refs=refs)

    # remove references from text
    if remove_refs:
        if refs is None:
            dat = remove_reference_section(dat)
        else:
            dat = remove_reference_section(dat, refs=refs)

    dat.sort_index(inplace=True)
    return dat

# Export loaded data
# raw = get_data(add_refs=False, remove_refs=False)
dat = get_data()
