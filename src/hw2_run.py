#!/usr/bin/env python

"""
Main driver program to fit some LDA models.
"""

from hw2_config import *
from hw2_tokenize import build_tokenizer, tokenizer_info
from hw2_data import NipsData
from hw2_lda import run_lda, print_top_words

# Load data -- references removed and added as new column by default
nips = NipsData()
dat = nips.load_data()
top_n_words = 10                # number of words to print / topic

# Fit LDA model with bigrams on abstracts
lda_abs, bigrams_abs, bi_tokenizer_abs =\
    run_lda(dat.abstract, msg="LDA w/ bigrams on abstracts", ngram_range=(1, 2),
            n_components=20) 

print_top_words(lda_abs, bi_tokenizer_abs.get_feature_names(), top_n_words)

# Fit LDA w/ bigrams on texts
lda_txt, bigrams_txt, bi_tokenizer_txt =\
    run_lda(dat.text, msg="LDA w/ bigrams on texts", ngram_range=(1, 2),
            n_components=20) 

print_top_words(lda_txt, bi_tokenizer_txt.get_feature_names(), top_n_words)
