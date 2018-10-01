#!/usr/bin/env python

"""
Driver to run LDA models. Work in progress. But the topics output seem pretty
decent.
"""

import numpy as np
import time

# LDA / NMF models
from sklearn.decomposition import NMF, LatentDirichletAllocation

from hw2_config import *
from hw2_tokenize import *
from hw2_data import NipsData

# LDA variables
n_components = 20
n_top_words = 10

# http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
def print_top_words(model, feature_names, n_top_words):
    print()
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ' '.join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
# Load data -- references removed and added as new column by default
nips = NipsData()
nips.load_data()
dat = nips.data

## -------------------------------------------------------------------
### LDA models with unigrams and bigrams
# tokenize
tf_uni_tokenizer = build_tokenizer(count=True)
tf_unigrams = tf_uni_tokenizer.fit_transform(dat.text)
tokenizer_info("TF unigram", tf_unigrams, tf_uni_tokenizer)

lda = LatentDirichletAllocation(n_components=n_components, random_state=RANDOM_SEED)
print("Fitting LDA on TF unigrams")
t0 = time()
lda.fit(tf_unigrams)
print(f"done in {(time() - t0):0.3f} secs.")
print_top_words(lda, tf_uni_tokenizer.get_feature_names(), n_top_words)

# bigrams
tf_bi_tokenizer = build_tokenizer(count=True, ngram_range=(1, 2))
tf_bigrams = tf_bi_tokenizer.fit_transform(dat.text)
lda2 = LatentDirichletAllocation(n_components=n_components, random_state=RANDOM_SEED)

print("Fitting LDA on TF bigrams")
# t0 = time()
lda2.fit(tf_bigrams)
# print(f"done in {(time() - t0):0.3f} secs.")
print_top_words(lda2, tf_bi_tokenizer.get_feature_names(), n_top_words)
