#!/usr/bin/env python

"""
Functions for running LDA models
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# LDA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

from hw2_config import *
from hw2_utils import *
from hw2_tokenize import build_tokenizer


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
        print("Building tokenizer...")
        tokenizer = build_tokenizer(count=True, ngram_range=ngrams)
    print("Tokenizing...")
    tokens = kw.pop('tokens', tokenizer.fit_transform(data))
    
    # LDA variables -- these are overriden by passed params
    lda_defaults = dict(
        n_components=20,
        random_state=RANDOM_SEED
    )
    args = {**lda_defaults, **kw}

    # Run LDA
    lda = LatentDirichletAllocation(**args)
    print("Fitting model...")
    lda.fit(tokens)

    return (lda, tokens, tokenizer)


def run_lda_all_sections(nips, **kw):
    """Run LDA on abstract, title, text, and combinations of the three
    using both unigrams and bigrams."""
    n_components = kw.pop('n_components', 20)
    objs = pickle_load('models.pkl') or dict()
    colloc = kw.pop("colloc", [])      # join collocated bigrams or trigrams
    
    for section in ["title", "abstract", "text",
                    ["title", "abstract"],
                    ["title", "text"],
                    ["title", "abstract", "text"]]:
        for ngram in ["uni", "bi"]:
            for c in colloc:
                key = '_'.join(['lda', str(n_components), ngram, *section])
            if joins:
                key += '_' + joins
            ng = (1, 1) if ngram is "uni" else (1, 2)
            # Note: when using multiple sections, the sections must be part
            # of the same document
            dat = nips.raw[section] if isinstance(section, str) else \
                nips.raw[section].apply(lambda x: '\n'.join(x), axis=1)

            if objs.get(key) is None:
                objs[key] =\
                    run_lda(dat,
                            msg=f"Running LDA({ngram}gram) on {section}",
                            n_components=n_components, ngram_range=ng)
                pickle_save(objs, 'models.pkl') # save every time in case interrupted

            
## -------------------------------------------------------------------
### Evaluation

def plot_lda_topic_barplot(model, topic, feature_names, n_words, **kw):
    """Plot barplot of weights of top n_words in topic."""
    inds = model.components_[topic].argsort()[:-n_words - 1:-1]
    sns.barplot(x=np.array(feature_names)[inds], y=model.components_[topic][inds],
                palette='Blues_d')
    plt.title(f"Topic #{topic}: top {n_words} words")
    plt.xlabel('Words')
    plt.ylabel('Weights')
    plt.show()


# http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
def print_top_words(model, feature_names, n_top_words):
    print()
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ' '.join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
                
