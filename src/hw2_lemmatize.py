#!/usr/bin/env python

# Install NLTK lemma stuff

# NP: I've been playing around with the WordNetLemmatizer a bit and it seems
#     pretty good with wordnet.
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import PorterStemmer

from hw2_utils import *
from hw2_config import *

## -------------------------------------------------------------------
### Bag of words / stemming / lemmatization

# if USE_LEMMA:                   # use wordnet's lemmatization dictionary
#     nltk.data.path = [rootdir]
#     nltk.download('wordnet', download_dir=rootdir)

# FIXME: WIP
def get_dict(grams, stem=USE_STEM, lemma=USE_LEMMA):
    """Convert text to BOWs, optionally applying a stemmer."""
    words = grams.get_feature_names()
    if lemma:
        words = WordNetLemmatizer().lemmatize(text, pos='v')
    if stem:
        # FIXME: stem or not to stem?
        stemmer = SnowballStemmer('english')

def main():
    """Just intalls wordnet lemma corpus when called as script."""
    # downloads lemma corpora -- not too big
    nltk.download("wordnet", download_dir=root_path())
    
if __name__ == '__main__':
    main()
