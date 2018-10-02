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

## Text pipeline & NLP packages
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
# from nltk.stem.porter import PorterStemmer

from hw2_config import *        # project constants
from hw2_utils import *         # helper functions

def get_nltk_prereqs():
    """Download NLTK prereqs in root directory."""
    nltk.download(['wordnet', 'punkt'],
                  download_dir=os.path.join(root_path(), NLTKDIR))

## -------------------------------------------------------------------
### Word preprocessing:
#   - split into lowercase words >3 chars, removing punctuation
#   - lemmatization if configured will 
#      - convert 3rd person -> 1st person
#      - convert past tense and future tense verbs to present
#   - stopwords are removed using scikit-learns defaults

# Lemmatizer: can be used as a replacement tokenizer + lemmatizer in
# scikit-learn tokenizers
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.rt = RegexpTokenizer('(?ui)\\b[a-z]{3,}\\w*\\b')

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in self.rt.tokenize(doc)]

# Tokenization defaults to TF/TF-IDF sparse matrices
def build_tokenizer(**args):
    """
    Return a TF or TF-IDF n-gram tokenizer using scikit-learn's
    TfidfVectorizer or CountVectorizer.
    
    Arguments: 
      - count: if count=True returns a CountVectorizer, otherwise TfidfVectorizer
      - all other arguments are passed on to the returned vectorizer.
    """
    cv = args.pop('count', False) # return CountVectorizer

    # These are default arguments for scikit-learn's TfidfVectorizer/CountVectorizer
    tokenizer_defaults = dict(
        lowercase=True,          # convert everything to lowercase
        decode_error='ignore',   # throw out unparseables
        strip_accents='unicode', # preprocessing
        stop_words='english',
        max_df=0.90,             # auto build ignored terms
        min_df=0.05,             # terms only appear in small fraction of docs
        tokenizer=LemmaTokenizer() if USE_LEMMA else None,
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


def tokenizer_info(name, grams, tokenizer):
    """Print some info about the tokenizer, include a few of the stopwords."""
    params = tokenizer.get_params()
    print(f"{name}\n--------------------")
    print(f"Shape: {grams.shape}")
    print(f"Words: {len(tokenizer.get_feature_names())}")
    if params['stop_words'] != None:
        print(f"Stopwords: {len(tokenizer.get_stop_words())}")
        print("First 10 stopwords:")
        pp.pprint(list(tokenizer.get_stop_words())[:10])
    else:
        print(f"Stopwords auto (min_df, max_df): \
({params['min_df']}, {params['max_df']})")
