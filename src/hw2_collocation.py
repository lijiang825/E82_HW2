#!/usr/bin/env python

"""
Find statistically collocated N-grams, eg. 'neural network'.
"""
import pandas as pd
import numpy as np

from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.corpus import stopwords

from hw2_config import *
from hw2_tokenizers import NgramPreTokenizer


def collocated_ngrams(data, num=100, kind='bigram', **kw):
    """
    Find NUM best bi/trigrams using wordnet's collocator. 'best' is determined
    by wordnets likelihood metric. KIND can be 'bigram' or 'trigram'.

    Returns list of tuples of ngrams.
    """
    if kind not in ['bigram', 'trigram']:
        raise ValueError("kind must be 'bigram' or 'trigram'")

    # the collocators don't seem to care if stopwords were previously removed
    stopset = set(stopwords.words('english'))
    npt = NgramPreTokenizer(**kw)
    tokens = data.map(npt)                          # tokenize the data

    cf = BigramCollocationFinder if kind is 'bigram' else TrigramCollocationFinder
    cf = cf.from_documents(tokens)
    cf.apply_word_filter(lambda w: w in stopset)         # remove grams w/ stopwords
    measure = BigramAssocMeasures if kind is 'bigram' else TrigramAssocMeasures
    grams = cf.nbest(measure.likelihood_ratio, num) # find the best
    
    return grams


# This can take quite a while running on the entire dataset, especially
# for trigrams. It uses NLTK for the collocations, doesn't seem to be multi-threaded
# like the scikit-learn functions.
@timeit
def compute_collocated_ngrams(data, num=10, ngram='bigram'):
    """Compute top NUM collocated n-grams for title, abstract, and text sections.
    N-grams can be 'bigrams' or 'trigrams'. Returns a dataframe of results."""
    return pd.DataFrame({
        "title": collocated_ngrams(data.title, num, ngram),
        "abstract": collocated_ngrams(data.abstract, num, ngram),
        "text": collocated_ngrams(data.text, num, ngram)
    })


def compute_all_collocations(nips, objs=None):
    """Compute all the collocated data that isn't pickled and store it."""
    df = nips.load_data()
    if objs is None:
        print("Loading collocs pickle")
        objs = pickle_load('collocs.pkl') or dict()

    if objs.get("all_bgs") is None: # these both take a while
        print("Computing collocated bigrams on entire dataset...")
        objs["all_bgs"] =\
            compute_collocated_ngrams(nips.raw, 500, 'bigram')
        pickle_save(objs, 'collocs.pkl')
    if objs.get("all_tgs") is None:
        print("Computing collocated trigrams on entire dataset...")
        objs["all_tgs"] =\
            compute_collocated_ngrams(nips.raw, 500, 'trigram')
        pickle_save(objs, 'collocs.pkl')
    if objs.get("bgs_by_yr") is None: # top 10 / year
        print("Computing collocated bigrams/year...")
        objs["bgs_by_yr"] =\
            nips.raw.groupby('year')\
                    .apply(compute_collocated_ngrams)\
                    .reset_index(drop=1)
        pickle_save(objs, 'collocs.pkl')
    if objs.get("tgs_by_yr") is None: # top 10 / year
        print("Computing collocated trigrams/year...")
        objs["tgs_by_yr"] =\
            nips.raw.groupby('year')\
                    .apply(lambda x: compute_collocated_ngrams(x, ngram='trigram'))\
                    .reset_index(drop=1)

    if objs.get('bgs') is None: # union of common bgs / year
        objs['bgs'] =\
            list(filter(lambda t: '' not in t,
                        np.unique(objs['bgs_by_yr'].unstack())))
        
    if objs.get('tgs') is None: # union of common tris / year
        objs['tgs'] =\
            list(filter(lambda t: '' not in t,
                        np.unique(objs['tgs_by_yr'].unstack())))
        
    assert len(objs) > 0
    print("Saving collocs pickle")
    pickle_save(objs, 'collocs.pkl')
    return objs


# Overlap between bigrams from different sections
# Highest overlap b/w text/abstract and abstract/title
def print_collocated_ngrams(df):
    """Print info about commonality b/w common n-grams in different document 
    sections."""
    tmp = df.apply(set)
    ngram = f"{len(df.iloc[0, 1])}-gram"
    print()
    print(f"Overlap in {ngram}s b/w doc sections:")
    print("  - title/text:\t\t", len(tmp.title & tmp.text))
    print("  - title/abstact:\t", len(tmp.title & tmp.abstract))
    print("  - text/abstract:\t", len(tmp.text & tmp.abstract))
    print("  - title/text/abstact:\t", len(tmp.title & tmp.abstract & tmp.text))
    print()
    print(df.head(5))
