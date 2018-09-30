#!/usr/bin/env python

# NP: The data I'm using appears to have a few more entries than the dataset you
#     are using after tokenization

# This script just contains code for the tokenization
# NP: The lemmatization is a WIP and I was unsure on stemming so I was trying to
# make them both configurable -- see hw2_config.py

import pandas as pd
import numpy as np

## Text pipeline & NLP packages
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from hw2_config import *        # project constants
from hw2_utils import *         # helper functions
from hw2_data import dat        # data to use

## -------------------------------------------------------------------
### Preliminary: tokenization, normalization

# Tokenization to TF/TF-IDF sparse matrices
# These are default arguments for scikit-learn's TfidfVectorizer
tokenizer_defaults = dict(
    lowercase=True,          # convert everything to lowercase
    decode_error='ignore',   # throw out unparseables
    strip_accents='unicode', # preprocessing
    stop_words='english',
    max_df=0.95,             # auto build ignored terms
    # min_df=0.02,           # this seems unnecessary? but may reduce memory
    norm='l2',               # normalize term vectors
    # I think we can just set this to False to have TF only tokens?
    # use_idf=True,          # use inverse-document-freq. weighting
    smooth_idf=True,         # adds 1 to avoid division by zero errors for tf-idf
    # I modified this slightly since there were lots of numbers in the tokens
    token_pattern=u'(?ui)\\b[a-z]+\\w*\\b' # ignore words starting with numbers
)

def build_tokenizer(**args):
    """Return a TF or TF-IDF n-gram tokenizer."""
    return TfidfVectorizer(**args, **tokenizer_defaults)

# build some tokenizers
tfidf_unigram_tokenizer = build_tokenizer()
tfidf_bigram_tokenizer = build_tokenizer(ngram_range=(1, 2)) # or (2, 2) ?
tf_unigram_tokenizer = build_tokenizer(use_idf=False)
tf_bigram_tokenizer = build_tokenizer(use_idf=False, ngram_range=(1, 2))

# Tokenize the data into TF and TF-IDF tokenized, L2 normalized, and lowercase
# NOTE:
# - we could add in stemming and lemmatization as well? 
# - I think it may well be worth adding in at least lemmatization, the
#   stemming seems to make things a little harder to interpret, but may be
#   good as well - I haven't messed around with it too much yet.
# - This just tokenizes the text
tf_unigrams = tf_unigram_tokenizer.fit_transform(dat['text'].values)
tf_bigrams = tf_bigram_tokenizer.fit_transform(dat['text'].values)
tfidf_unigrams = tfidf_unigram_tokenizer.fit_transform(dat['text'].values)
tfidf_bigrams = tfidf_bigram_tokenizer.fit_transform(dat['text'].values)

def tokenizer_info(name, grams, tokenizer):
    """Print some info about the tokenizer, include a few of the stopwords."""
    print(f"{name}\n--------------------")
    print(f"Shape: {grams.shape}")
    print(f"Words: {len(tokenizer.get_feature_names())}")
    print(f"Stopwords: {len(tokenizer.get_stop_words())}")
    print("First 10 stopwords:")
    print(list(tokenizer.get_stop_words())[:10])
    
# info on tokens
tokenizer_info("TF-IDF Unigrams", tfidf_unigrams, tfidf_unigram_tokenizer)
tokenizer_info("TF-IDF Bigrams", tfidf_bigrams, tfidf_bigram_tokenizer)
tokenizer_info("TF Unigrams", tf_unigrams, tf_unigram_tokenizer)
tokenizer_info("TF Bigrams", tf_bigrams, tf_bigram_tokenizer)
