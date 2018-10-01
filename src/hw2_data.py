#!/usr/bin/env python

"""
This module contains a wrapper class to store the NIPs data.
Configuration can be handled in hw2_config by setting global constants,
or in the class constructor.

By default, when data is loaded it: 
  - creates lists of references from the text of the articles (from the 
    References section), adding a new column "refs" to the data.
  - removes the Reference section from the text
  - takes a working sample of the data if requested

Useful methods:
  - load_data: loads data using the current settings
  - resample: resamples the raw dataset a given sample fraction
  - print_config: print out the instance configuration

Note: when left with defaults, the final columns will be 
  ['id', 'year', 'title', 'abstract', 'text', 'refs']
and the shape (7241, 6).
"""
# NP: I have been storing data in ./data/nips-papers.xlsx

# FIXME: where to store shared dataset?
# NP: the data I'm using here is the cleaned up version someone posted on piazza
# Here is the google drive link:
# 
# https://drive.google.com/file/d/180FBOXqxdyvzHihHsg_bAbYS-UF1WrKZ/view?usp=sharing
#
# It would be nice to be able to access shared dataset from our scripts, but
# I haven't looked into the python google-drive API (there is one though). Is
# there an alternative way for us to store our data?

import pandas as pd
import numpy as np
import os

from hw2_utils import *         # helper functions: root_path
from hw2_config import *        # project constants

class NipsData:
    """Wrapper around NIPs dataset."""
    raw = None                  # raw data
    data = None                 # resulting data after transforms
    refs = None                 # extracted references
    has_refs = True             # data has a 'refs' column
    refs_removed = True         # reference section is removed from text
    root = None                 # project root
    data = None                 # data file
    datadir = None              # data directory
    is_sample = True            # use sample of total dataset
    sample_frac = None          # fraction of data to use in sample

    def __init__(self, **kwargs):
        """
        Manages NIPs data.
        Default options (can be passed in constructor or changed in global config):
          - add column with list of references extracted from text
          - remove the references section from article text
          - use a sample of the raw data
        """
        self.has_refs = kwargs['add_refs'] if 'add_refs' in kwargs else EXTRACT_REFS
        self.refs_removed = kwargs['remove_refs'] if 'remove_refs' in kwargs \
            else REMOVE_REFS
        self.root = kwargs['root'] if 'root' in kwargs else root_path()
        self.datafile = kwargs['datafile'] if 'datafile' in kwargs else DATAFILE
        self.datadir = kwargs['datadir'] if 'datadir' in kwargs else DATADIR
        self.is_sample = kwargs['sample'] if 'sample' in kwargs else SAMPLE
        self.sample_frac = kwargs['sample_frac'] if 'sample_frac' in kwargs else \
            SAMPLE_FRAC
        self.datapath = kwargs['datapath'] if 'datapath' in kwargs \
            else os.path.join(self.root, self.datadir, self.datafile)


    def print_config(self):
        """Print this instances configuration."""
        print(f"Data: {self.datapath}")
        print(f"Refs: {self.has_refs}")
        print(f"Refs-Removed: {self.refs_removed}")
        print(f"Sample: {self.is_sample}")
        if (self.is_sample):
            print(f"Sample-Fraction: {self.sample_frac}")
        

    def load_data(self, **kwargs):
        """
        Loads data. Data location is specified in class constructor or set 
        manually.
        """
        self.raw = pd.read_excel(self.datapath) if self.raw is None else self.raw
        
        # Use only a fraction of data for preliminary runs
        if self.is_sample:
            seed = kwargs['seed'] if 'seed' in kwargs else RANDOM_SEED
            self.data = self.raw.sample(frac=self.sample_frac, random_state=seed)
        else:
            self.data = self.raw.copy(deep=True)
            
        # find references if necessary
        if self.has_refs or self.refs_removed:
            self.find_references()
            
        # add references to data
        if self.has_refs:
            self.add_reference_column()

        # remove references from text
        if self.refs_removed:
            self.remove_reference_section()

        self.data.sort_index(inplace=True)


    def resample(self, fraction, **kwargs):
        """Resample data using FRACTION of total and update references, etc."""
        self.sample_frac = fraction
        seed = kwargs["seed"] if "seed" in kwargs else np.random.randint(0, 1000)
        kwargs = dict(seed = seed)
        self.load_data(**kwargs)
        
    ## -------------------------------------------------------------------
    ### Managing References

    def find_references(self):
        """
        Pulls out references from papers, returns a list of references indexed by
        paper ID. Only returns references from papers where a reference section was
        actually found. 

        Notes: 
          - some papers are truncated / missing references section. 
          - Also the references appear to be truncated in cases (eg. 6556).
          - the text is split by a reference regex that may match in multiple
            places, so the last element is the correct one. However, the entire
            list is returned here to enable later processing if desired (eg.
            removing the reference section).

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
        self.refs = self.data.text.str.split(
            r"\s+[Rr][Ee][Ff][Ee][Rr][Ee][Nn][Cc][Ee][Ss]\s+(?=[\[\(][0-9]{1,3}|[0-9]{1,3}[.]\s+)")
        mask = (self.refs.apply(len) > 1) # ignore texts with no detected references
        self.refs = self.refs[self.data.text[mask].index]


    def split_references(self, refs):
        """
        Split reference text into references where possible.

        They could be of forms: [1] ..., (1) ..., or 1. ...
        Others are ignored (hopefully).

        There are also assumed to only be upto 3 digits in a reference to avoid 
        possible complication with years.
        """
        return refs.str.split(
            r"(?:^|\A|\n)\s*(?:[\[\(][0-9]{1,3}[\]\)]|[0-9]{1,3}[.]\s+)\s*")\
                       .apply(lambda x: [i for i in x if len(i) > 0])


    def add_reference_column(self):
        """Add references to data. Pass pre-computed 'refs' to avoid recomputation."""
        if 'refs' in self.data:
            print("Already have a references column: 'refs'")
            return 
        # Only use the last section
        self.data["refs"] = self.split_references(self.refs.apply(lambda x: (x[-1:])[0]))


    def remove_reference_section(self):
        """
        Remove the reference section from text. The text is split into possible 
        reference sections by `find_references`, and the last section, presumably the
        actual reference section is removed.

        The split sections, minus the last one, are joined back together with 
        ' references '.
        """
        # a couple have random refs, but the last section should be right
        self.data.text[self.refs.index] = \
            self.refs.apply(lambda x: ' references '.join(x[:-1]))
