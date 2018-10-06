#!/usr/bin/env python

"""
Tokenization:
  - lowercases
  - removes punctuation
  - cleans up unicode
  - breaks strings into words of at least 3 letters using RegexpTokenizer
  - lemmatizes words using WordNetLemmatizer if USE_LEMMA is True
  - stopwords are computed on the fly using min_df and max_df params

Normalization:
  - generates CountVectorizers, TF or TF-IDF tokenizers
  - TF are normalized using 'l2' norm
  - TF-IDF are further normalized by frequency of word / doc to word globally

See hw2_config for configuration.

Note: to use the nltk lemmatizer, there are a couple downloads required. The
wrapper function 'get_nltk_prereqs' will install them in the project root 
directory.

Example to create a bigram CountVectorizer:

    tokenizer = build_tokenizer(count=True, ngram_range=(1, 2))
    tokenizer.fit_transform(data)
    tokenizer_info("Unigram tokenizer", tokens, tokenizer)
"""

import pandas as pd
import numpy as np

## Text pipeline & NLP packages
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from hw2_tokenizers import LemmaTokenizer, NgramLemmaTokenizer
from hw2_config import *

## -------------------------------------------------------------------
### Word preprocessing:
#   - split into lowercase words >3 chars, removing punctuation
#   - by default lemmatizes nouns, verbs, adjectives, and adverbs
#     eg,
#      - unpluralizes nouns
#      - converts 3rd person -> 1st person
#      - converts past tense and future tense verbs to present
#      - etc, see WordNetLemmatizer
#   - stopwords are removed using scikit-learns defaults + rare and common terms
#   - optionally join most common bigrams/trigrams into single entities

# Tokenization defaults to TF/TF-IDF sparse matrices
def build_tokenizer(**args):
    """
    Return a TF or TF-IDF n-gram tokenizer using scikit-learn's
    TfidfVectorizer or CountVectorizer.
    
    Arguments: 
      - count: if count=True returns a CountVectorizer, otherwise TfidfVectorizer
      - pos: passed to LemmaTokenizer (part of speech - default nouns)
      - all other arguments are passed on to the returned vectorizer, overriding
        defaults.
    """
    cv = args.pop('count', False) # return CountVectorizer
    pos = args.pop('pos', None)   # defaults - nouns, adverbs, verbs, adjs
    
    # These are default arguments for scikit-learn's TfidfVectorizer/CountVectorizer
    tokenizer_defaults = dict(
        lowercase=True,          # convert everything to lowercase
        decode_error='ignore',   # throw out unparseables
        strip_accents='unicode', # preprocessing
        stop_words='english',
        max_df=0.75,             # auto build ignored terms
        min_df=2,                # terms only appear in small fraction of docs
        max_features=5000,       # vocab limited to this size
        tokenizer=LemmaTokenizer(pos=pos) if USE_LEMMA else None,
        # I modified this slightly since there were lots of numbers in the tokens
        # also allows words joined by '-' or '_'
        token_pattern=u'(?ui)\\b[A-Za-z][A-Za-z0-9_-]+[A-Za-z0-9]{1,}\\b'
    )

    tfidf_defaults = dict(
        norm='l2',       # normalize term vectors
        use_idf=True,    # default: True - use inverse-document-freq. weighting
        smooth_idf=True, # adds 1 to avoid division by zero errors for tf-idf
    )
    tfidf_defaults = {**tfidf_defaults, **tokenizer_defaults}

    # passed arguments override the defaults
    if cv:
        return CountVectorizer(**{**tokenizer_defaults, **args})
    return TfidfVectorizer(**{**tfidf_defaults, **args})


def build_joined_tokenizer(ngrams, **kw):
    """Build tokenizer with joined ngrams, eg. bigrams or trigrams."""
    pos = kw.pop('pos', None)    # lemmatize param
    return build_tokenizer(
        tokenizer=NgramLemmaTokenizer(ngrams=ngrams, pos=pos), **kw)


def tokenize_lemmatize(data, keep_joins=False, **kw):
    """Tokenize, lemmatize, and optionally stem. keep_joins keeps [-_].
    Optional args: 
      stem=True: use PorterStemmer to stem words
      translate=True: convert '-' to '_'
    """
    return data.apply(text_preprocess)\
               .apply(GensimTokenizer(translate=keep_joins))


def tokenizer_info(name, grams, tokenizer):
    """Print some info about the tokenizer, include a few of the stopwords."""
    top10 = np.argsort(-np.sum(grams.toarray(), axis=0))[:10]
    params = tokenizer.get_params()
    dense = grams.todense()
    swords = len(tokenizer.stop_words_)
    print(f"{name}\n--------------------")
    print(f"Sparsity: {((dense > 0).sum() / dense.size)*100:0.3f}%")
    print(f"Shape: {grams.shape}")
    print(f"Words: {len(tokenizer.get_feature_names())}")
    print("Top 10 words by count:")
    pp.pprint(list(np.array(tokenizer.get_feature_names())[top10]))
    if params['stop_words'] != None:
        swords += len(params['stop_words'])
    print("--------------------")
    print(f"Stopwords: {swords}")
    print("First 10 stopwords:")
    pp.pprint(list(tokenizer.get_stop_words())[:10])
    print()
    print(f"Stopwords auto (min_df, max_df): \
({params['min_df']}, {params['max_df']})")
    print("--------------------")
    print("Tokenizer:")
    print()
    print(tokenizer)
