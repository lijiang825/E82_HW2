#!/usr/bin/env python

"""
Main driver program to fit some LDA models.
"""
import pandas as pd
import numpy as np

from hw2_config import *
from hw2_data import NipsData
from hw2_tokenize import *
from hw2_collocation import *
from hw2_lda import *
from hw2_model_analysis import *
from hw2_lda_gensim import *

# Load data -- references removed and added as new column by default
nips = NipsData()
dat = nips.load_data(sample_frac=0.5)

# LDA settings
top_n_words = 10                # number of words to print / topic
n_components = 20               # topics for LDA

# Load pickled objects: this holds objects that take a long time to compute
cs = pickle_load('collocs.pkl') or dict()

## -------------------------------------------------------------------
### Collocated ngrams
# Note: this takes a long time to run, it can computed from a different
# shell with 'make all-data'
compute_all_collocations(nips, cs)
bgs, all_bgs, bgs_by_year = cs.get('bgs'), cs.get("all_bgs"), cs.get('bgs_by_yr')
tgs, all_tgs, tgs_by_year = cs.get('tgs'), cs.get('all_tgs'), cs.get('tgs_by_yr')
    
# collocated ngrams using entire dataset: highest overlap b/w text and abstracts
# this is badly skewed towards recent years
print_collocated_ngrams(all_bgs)
print_collocated_ngrams(all_tgs)

# Nearly every trigrams contains "neural", they are all related to neural networks
# Bigrams are much more variable
pd.DataFrame(np.mean(all_tgs.applymap(lambda x: "neural" in x)), columns=["%neural"])
pd.DataFrame(np.mean(all_bgs.applymap(lambda x: "neural" in x)), columns=["%neural"])

# However, using top ngrams / year shows different trends
# There is a big gap in neural network trigram popularity from late 1999-2014
nnet = tgs_by_year.groupby(tgs_by_year.index)\
                   .agg(lambda c: len([elem for elem in c
                  if any([i in elem for i in ['neural', 'network']])]))
sns.heatmap(nnet.sort_index(ascending=False), cmap='Reds')
plt.title("Presence of 'neural' or 'network' in Top 10 Trigrams\nOver the Years")
plt.show()

# pairs common to all sections
cbgs_intersect = set(all_bgs.text) & set(all_bgs.abstract) & set(all_bgs.title)
ctgs_intersect = set(all_tgs.text) & set(all_tgs.abstract) & set(all_tgs.title)

print("Bigrams common to all sections: ", len(cbgs_intersect))
pp.pprint(list(cbgs_intersect)[:10])
print("Trigrams common to all sections: ", len(ctgs_intersect))
pp.pprint(list(ctgs_intersect)[:10])

# The models use union of all sections from calculations / years
# Hopefully, this allows to see declining topics as well by reducing the
# skew
print("Union of common bigrams from all years: ", len(bgs))
print("Union of common trigrams from all years: ", len(tgs))

## -------------------------------------------------------------------
### LDA models

# optimization: check n_components
optim = pickle_load('lda_optim.pkl')
model = optim.get('lda_optim_title_text')

if not model:
    # Use subset of data
    dat = nips.resample(0.2)
    tst = nips.combined_sections(['title', 'abstract'])
    tf = build_joined_tokenizer(ngrams=common_bigrams)
    toks = tf.fit_transform(tst)
    # uses LatentDirichletAllocation.score function (basically log-likelihood)
    model = lda_optimize_model(
        toks, param_grid={'n_components': [10, 20, 30, 50]})

# However, the score function doesn't work well at all, so I
# abandonded trying to optimize and just went with a default of 10
# components for the LDA models.
# Note: I spent a while trying to adapt a version of the CoherenceModel
# to scikit-learn but to no avail
# just constantly increaing w/ fewer topics
print_lda_optim_results(model, sections = ["title", "text"])
plot_lda_optim_topics(
    model, title='Optimization of LDA Topic Number using GridSearch')

## -------------------------------------------------------------------
### Models
mods = pickle_load('models.pkl')

# Runs LDA with unigrams/bigrams on abstract, title, text, and all three
# combined. If these models are already pickled, then nothing is run.
# run_lda_all_sections(nips, n_components=n_components)
dat = nips.resample(0.2)
tst = nips.combined_sections(['title', 'text'])

tf = build_tokenizer(count=True)
toks = tf.fit_transform(dat.title)


tf = build_joined_tokenizer(count=True, ngrams=common_bigrams)
toks = tf.fit_transform(dat.abstract)

tf = build_joined_tokenizer(count=True, ngrams=common_trigrams)
toks = tf.fit_transform(dat.title)


# model run using bigrams on all sections
# model, grams, tk = mods.get('lda_bi_title_abstract')
# print_top_words(model, tk.get_feature_names(), top_n_words)
# plot_lda_topic_barplot(model, 18, tk.get_feature_names(), 5)
