#!/usr/bin/env python

"""
LDA models. Work in progress. But the topics output seem pretty decent.
TODO: account for more than just unigram topics???
"""

# LDA / NMF models
from sklearn.decomposition import NMF, LatentDirichletAllocation

from hw2_config import *
from hw2_tokenize import build_tokenizer

# http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
def print_top_words(model, feature_names, n_top_words):
    print()
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ' '.join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    

@timeit
def run_lda(data, **kw):
    """
    Run LDA on data. Unless a tokenizer is passed as a parameter, it constructs
    a CountVectorizer using build_tokenizer.

    Optional arguments:
      - msg: print a message before run starts
      - ngram_range: passed to CountVectorizer
      - tokenizer: tokenizer to use
      - all other arguments are passed to LatentDirichletAllocation
    
    Returns (model, tokens, tokenizer)
    """
    print()
    if 'msg' in kw:
        print(kw.pop('msg', None))

    # construct n-grams
    ngrams = kw.pop('ngram_range', (1, 1))
    if 'tokenizer' in kw:
        tokenizer = kw.pop('tokenizer', None)
    else:
        tokenizer = build_tokenizer(count=True, ngram_range=ngrams)
        print("Tokenizing...")
    tokens = kw.pop('tokens', tokenizer.fit_transform(data))
    
    # LDA variables -- these are overriden by passed params
    lda_defaults = dict(
        n_components=50,
        random_state=RANDOM_SEED
    )
    args = {**lda_defaults, **kw}

    # Run LDA
    lda = LatentDirichletAllocation(**args)
    print("Fitting model...")
    lda.fit(tokens)

    return (lda, tokens, tokenizer)
