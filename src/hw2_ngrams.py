#!/usr/bin/env python

"""
WIP: Attempt to construct bigrams/trigrams using collocated words.
"""

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.corpus import stopwords
import string
import regex as re
import unicodedata
import sys

from hw2_tokenize import LemmaTokenizer


class NgramLemmaTokenizer(LemmaTokenizer):
    """
    Works like the LemmaTokenizer, but also deals with lowercasing,
    stopwords, and punctuation that are handled in the vectorizers.
    """
    def __init__(self):
        super().__init__()
        self.tbl = dict.fromkeys( # translation table to remove unicode/punct
            i for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith('P'))
        self.sw = stopwords.words('english')
        
    def __call__(self, doc):
        return [self.wnl.lemmatize(t.lower().translate(self.tbl))
                for t in self.rt.tokenize(doc) if t not in self.sw]


def join_bigrams(lst, ngrams):
    """Replace bigrams with their joined forms."""
    res = []
    skip = False
    for i in range(1, len(lst)):
        if skip:
            skip = False
            continue
        if (lst[i-1], lst[i]) in ngrams:
            res.append(lst[i-1] + '_' + lst[i])
            skip = True
        else:
            res.append(lst[i-1])
    return res


def construct_ngrams(data, num=1000, kind='bigram', insert=True):
    """
    Find bi/trigrams using wordnet's collocator. Create conjugates by combining
    words as 'w1_w2'. Finds the NUM best conjugates, determined using nltk's
    likelihood metric. KIND can be 'bigram' or 'trigram'.
    
    If INSERT is True, then a copy of the conjugate is inserted wherever a matching
    pair/triple is found. The pair/triples ARE removed to change the frequency 
    counts.

    Returns modified data if insert is True, otherwise resulting tuples
    """
    if kind not in ['bigram', 'trigram']:
        raise ValueError("kind must be 'bigram' or 'trigram'")

    nlt = NgramLemmaTokenizer()
    tokens = data.map(nlt)      # tokenize the data
    cf = BigramCollocationFinder if kind is 'bigram' else TrigramCollocationFinder
    cf = cf.from_documents(tokens)
    measure = BigramAssocMeasures if kind is 'bigram' else TrigramAssocMeasures
    grams = cf.nbest(measure.likelihood_ratio, num)
    
    if insert:
        # FIXME: assuming bigrams
        return tokens.map(lambda x: join_bigrams(x, grams))
    
    return grams
