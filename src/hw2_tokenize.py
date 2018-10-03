#!/usr/bin/env python

"""
Tokenization:
  - lowercases
  - removes punctuation
  - cleans up unicode
  - breaks strings into words of at least 3 letters using RegexpTokenizer
  - lemmatizes words (nouns) using WordNetLemmatizer if USE_LEMMA is True
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

## Text pipeline & NLP packages
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.corpus import stopwords
import string
import regex as re
import unicodedata
import sys
# from nltk.stem.porter import PorterStemmer

from hw2_config import *
from hw2_utils import *

## -------------------------------------------------------------------
### Word preprocessing:
#   - split into lowercase words >3 chars, removing punctuation
#   - lemmatization if configured will work on nouns (default)
#   - if set to verbs, pos='v',
#      - converts 3rd person -> 1st person
#      - converts past tense and future tense verbs to present
#   - stopwords are removed using scikit-learns defaults

# Lemmatizer: can be used as a replacement tokenizer + lemmatizer in
# scikit-learn tokenizers
class LemmaTokenizer(object):
    """
    Regexp tokenizer (words >= 3) and lemmatizer. Can be passed to 
    scikit-learn vectorizers.

    Optional arguments:
      - pos: passed to WordNetLemmatizer.lemmatize
    """
    def __init__(self, **kw):
        self.wnl = WordNetLemmatizer()
        self.rt = RegexpTokenizer('(?ui)\\b[a-z_]{3,}\\w*\\b')
        self.pos = kw.pop('pos', 'n') # lemmatize nouns by defaut
        
    def __call__(self, doc):
        return [self.wnl.lemmatize(t, pos=self.pos) for t in self.rt.tokenize(doc)]

    def __str__(self):
        return f"{type(self).__name__}(lemmatizer={self.wnl}, pos={self.pos},\n\t\
tokenizer={self.rt})"
        
    def __repr__(self):
        return self.__str__()


class NgramPreTokenizer(LemmaTokenizer):
    """
    Like NgramLemmaTokenizer but also does preprocessing:
      - converts to lowercase
      - removes unicode / punctuation
      - removes english stopwords
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.tbl = dict.fromkeys( # translation table to remove unicode/punct
            i for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith('P'))
        self.sw = stopwords.words('english')
        
    def __call__(self, doc):
        return [self.wnl.lemmatize(t.lower().translate(self.tbl), pos=self.pos)
                for t in self.rt.tokenize(doc) if t not in self.sw]


class NgramLemmaTokenizer(LemmaTokenizer):
    """
    Works like the LemmaTokenizer, but replaces collocated ngrams with joined
    versions, eg. 'neural network' => 'neural_network'.
    
    This attempts to address the BOW assumption that word locations are
    independent.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.ngrams = kw.pop('ngrams', None)
        
    def __call__(self, doc):
        res = [self.wnl.lemmatize(t, pos=self.pos) for t in self.rt.tokenize(doc)]
        return join_ngrams(res, self.ngrams) if self.ngrams is not None else res


def join_ngrams(lst, ngrams):
    """Replace ngrams with their joined forms, eg. 
    ['neural' 'network'] => ['neural_network']."""
    n = len(ngrams[0]) - 1
    res = []
    skips = 0
    for i in range(n, len(lst)):
        if skips > 0:
            skips -= 1
            continue
        if tuple(lst[(i-n):(i+1)]) in ngrams:
            res.append('_'.join(lst[(i-n):(i+1)]))
            skips += 1
        else:
            res.append(lst[i-n])
    return res
        

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
        print("Loading pickle")
        objs = pickle_load('collocs.pkl') or dict()

    if objs.get("common_bigrams") is None: # these both take a while
        print("Computing collocated bigrams on entire dataset...")
        objs["common_bigrams"] =\
            compute_collocated_ngrams(nips.raw, 100, 'bigram')

    if objs.get("common_trigrams") is None:
        print("Computing collocated trigrams on entire dataset...")
        objs["common_trigrams"] =\
            compute_collocated_ngrams(nips.raw, 100, 'trigram')

    if objs.get("common_bigrams_by_year") is None: # top 10 / year
        print("Computing collocated bigrams/year...")
        objs["common_bigrams_by_year"] =\
            nips.raw.groupby('year').apply(compute_collocated_ngrams)

    if objs.get("common_trigrams_by_year") is None:        # top 10 / year
        print("Computing collocated trigrams/year...")
        objs["common_trigrams_by_year"] =\
            nips.raw.groupby('year')\
                    .apply(lambda x: compute_collocated_ngrams(x, ngram='trigram'))
        
    print("Saving pickle -- Collocations done")
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
    pos = args.pop('pos', 'n')    # nouns are most useful, algorithms=>algorithm
    
    # These are default arguments for scikit-learn's TfidfVectorizer/CountVectorizer
    tokenizer_defaults = dict(
        lowercase=True,          # convert everything to lowercase
        decode_error='ignore',   # throw out unparseables
        strip_accents='unicode', # preprocessing
        stop_words='english',
        max_df=0.90,             # auto build ignored terms
        min_df=0.05,             # terms only appear in small fraction of docs
        tokenizer=LemmaTokenizer(pos=pos) if USE_LEMMA else None,
        # I modified this slightly since there were lots of numbers in the tokens
        token_pattern=u'(?ui)\\b[a-z]{3,}\\w*\\b' # ignore words starting with numbers
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
    """Build LDA tokenizer with joined ngrams, eg. bigrams or trigrams."""
    pos = kw.pop('pos', 'n')    # lemmatize param
    return build_tokenizer(
        count=True, tokenizer=NgramLemmaTokenizer(ngrams=ngrams, pos=pos), **kw)


def tokenizer_info(name, grams, tokenizer):
    """Print some info about the tokenizer, include a few of the stopwords."""
    params = tokenizer.get_params()
    dense = grams.todense()
    print(f"{name}\n--------------------")
    print(f"Sparsity: {((dense > 0).sum() / dense.size)*100:0.3f}%")
    print(f"Shape: {grams.shape}")
    print(f"Words: {len(tokenizer.get_feature_names())}")
    print("First 10 Words:")
    pp.pprint(list(tokenizer.get_feature_names()[:10]))
    if params['stop_words'] != None:
        print("--------------------")
        print(f"Stopwords: {len(tokenizer.get_stop_words())}")
        print("First 10 stopwords:")
        pp.pprint(list(tokenizer.get_stop_words())[:10])
    else:
        print(f"Stopwords auto (min_df, max_df): \
({params['min_df']}, {params['max_df']})")
    print("--------------------")
    print("Tokenizer:")
    print(tokenizer)
