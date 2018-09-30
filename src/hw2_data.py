#!/usr/bin/env python

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

import random
import pandas as pd
import numpy as np

from hw2_utils import *         # helper functions
from hw2_config import *        # project constants

rootdir = root_path()
datadir = os.path.join(rootdir, DATADIR)
datapath = os.path.join(datadir, DATA)
dat = pd.read_excel(datapath)

if PRELIM:                      # preliminary only uses a fraction of data
    random.seed(RANDOM_SEED)
    dat = dat.sample(frac=PRELIM_FRAC)
