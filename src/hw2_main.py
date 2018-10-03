#!/usr/bin/env python

"""
Main driver program to fit some LDA models.
"""
import pandas as pd
import numpy as np

from hw2_config import *
from hw2_utils import *
from hw2_data import NipsData
from hw2_lda import *

# Load data -- references removed and added as new column by default
nips = NipsData()
dat = nips.load_data(sample_frac=0.2)

# LDA settings
top_n_words = 10                # number of words to print / topic
n_components = 50               # topics for LDA

# Load pickled objects: this holds objects that take a long time to compute
cs = pickle_load('collocs.pkl') or dict()

## -------------------------------------------------------------------
### Collocated ngrams
# Note: this takes a long time to run, it can computed from a different
# shell with 'make all-data'
compute_all_collocations(nips, cs)
cbgs = cs.get("common_bigrams")
cbgs_by_year = cs.get("common_bigrams_by_year")
ctgs = cs.get("common_trigrams")
ctgs_by_year = cs.get("common_trigrams_by_year")
    
# collocated ngrams using entire dataset: highest overlap b/w text and abstracts
# this is badly skewed towards recent years
print_collocated_ngrams(cbgs)
print_collocated_ngrams(ctgs)

# Every trigrams contains "neural", they are all related to neural networks
# Bigrams are much more variable
print(np.alltrue(ctgs.applymap(lambda x: "neural" in x)))

# However, using top ngrams / year shows different trends
# There is a big gap in neural network trigram popularity from late 1999-2014
nnet = ctgs_by_year.groupby(ctgs_by_year.index)\
                   .agg(lambda c: len([elem for elem in c
                  if any([i in elem for i in ['neural', 'network']])]))
sns.heatmap(nnet.sort_index(ascending=False), cmap='Reds')
plt.title("Presence of 'neural' or 'network' in Top 10 Trigrams\nOver the Years")
plt.show()

# 

# Construct common ngrams to use in LDA
# pairs common to all sections
cbgs_intersect = set(cbgs.text) & set(cbgs.abstract) & set(cbgs.title)
ctgs_intersect = set(ctgs.text) & set(ctgs.abstract) & set(ctgs.title)

print("Bigrams common to all sections: ", len(cbgs_intersect))
pp.pprint(list(cbgs_intersect)[:10])
print("Trigrams common to all sections: ", len(ctgs_intersect))
pp.pprint(list(ctgs_intersect)[:10])

# Use union of all sections from calculations / years
# Hopefully, this allows to see declining topics as well by reducing the
# skew
common_bigrams = list(set(cbgs_by_year.title) | set(cbgs_by_year.text) \
                      | set(cbgs_by_year.abstract))
common_trigrams = list(set(ctgs_by_year.title) | set(ctgs_by_year.text) \
                       | set(ctgs_by_year.abstract))
print("Union of common bigrams from all years: ", len(common_bigrams))
print("Union of common trigrams from all years: ", len(common_trigrams))

## -------------------------------------------------------------------
### LDA models

# Runs LDA with unigrams/bigrams on abstract, title, text, and all three
# combined. If these models are already pickled, then nothing is run.
run_lda_all_sections(nips, n_components=n_components)

# model run using bigrams on all sections
model, grams, tk = objs.get('lda_bi_title_abstract')
print_top_words(model, tk.get_feature_names(), top_n_words)
plot_lda_topic_barplot(model, 18, tk.get_feature_names(), 5)
