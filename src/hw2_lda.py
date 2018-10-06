#!/usr/bin/env python

"""
Functions for running scikit-learn LDA models
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# LDA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, log_loss

from hw2_config import *
from hw2_tokenize import build_tokenizer, build_joined_tokenizer

# LDA variables -- these are overriden by passed params
lda_defaults = dict(
    n_components=20,
    random_state=RANDOM_SEED,
    max_doc_update_iter=100,
    max_iter=100,
    n_jobs=4,               # use more cores
    batch_size=1028         # larger batch size hopefully speeds up
)

def default_lda(**kw):
    """Return lda initialized with defaults (overridden by params)."""
    args = {**lda_defaults, **kw}
    return LatentDirichletAllocation(**args)
    
    
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
    tokens = kw.pop('tokens', None)
    print()
    if 'msg' in kw:
        print(kw.pop('msg', None))

    # construct n-grams
    max_df = kw.pop('max_df', 0.5)
    ngrams = kw.pop('ngram_range', (1, 1))
    tokenizer = kw.pop('tokenizer', None)
    if tokenizer is None:
        print("Building tokenizer...")
        tokenizer = build_tokenizer(count=True, ngram_range=ngrams, max_df=max_df)

    if tokens is None:
        print("Tokenizing...")
        tokens = tokenizer.fit_transform(data)

    # Run LDA
    lda = default_lda(**kw)
    print("Fitting model...")
    lda.fit(tokens)

    return (lda, tokens, tokenizer)


@timeit
def run_lda_all_models(nips, sections, **kw):
    """Run LDA on abstract, title, text, and combinations of the three
    using both unigrams and bigrams."""
    n_components = kw.pop('n_components', 20)
    mods = pickle_load('models.pkl') or dict()
    joins = kw.pop("colloc", False)      # join collocated bigrams or trigrams
    bgs = False
    tgs = False

    if joins:
        cs = pickle_load('collocs.pkl')
        bgs = True
        # tgs = True
        bgs = cs['bgs']         # union of common bigrams / year
        # tgs = cs['tgs']         # same w/ trigrams

    for section in sections:
        sec_str = '_'.join(section) if isinstance(section, list) else section
        # Note: when using multiple sections, the sections must be part
        # of the same document
        dat = nips.raw[section] if isinstance(section, str) else \
            nips.raw[section].apply(lambda x: '\n'.join(x), axis=1)

        if bgs:
            key = '_'.join(['lda', str(n_components), 'joined_bigram', sec_str])
            if mods.get(bkey) is None:
                tk = build_joined_tokenizer(ngrams=bgs)
                mods[key] =\
                    run_lda(dat,
                            msg=f"Running LDA(joined bgram) on {section}",
                            n_components=n_components, tokenizer=tk)
            pickle_save(mods, 'models.pkl')

        if tgs:
            key = '_'.join(['lda', str(n_components), 'joined_trigram', sec_str])
            if mods.get(tkey) is None:
                tk = build_joined_tokenizer(ngrams=tgs)
                mods[key] =\
                    run_lda(dat,
                            msg=f"Running LDA(joined trigrams) on {section}",
                            n_components=n_components, tokenizer=tk)
            pickle_save(mods, 'models.pkl')
            
        for ngram in ["uni", "bi"]: # regular unigram / trigram
            key = '_'.join(['lda', str(n_components), ngram, sec_str])
            ng = (1, 1) if ngram is "uni" else (1, 2)

            if mods.get(key) is None:
                mods[key] =\
                    run_lda(dat,
                            msg=f"Running LDA({ngram}gram) on {section}",
                            n_components=n_components, ngram_range=ng)
                pickle_save(mods, 'models.pkl') # save every time in case interrupted


## -------------------------------------------------------------------
### Model Selection

# This can take a while depending on the data size and the number of
# parameters to test
@timeit
def lda_optimize_model(data, param_grid, **kw):
    """Optimize LDA model on data using combination of params.
    param_grid should be a dictionary of mappings from parameter names to 
    possible values."""
    lda = default_lda()
    mod = GridSearchCV(lda, param_grid=param_grid, **kw)
    mod.fit(data)
    return mod


@timeit
def lda_optim_cli(data, **kw):
    """Run optimization non-interactively, since it takes a long time."""
    grid = kw.pop("pgrid", {'n_components': [10, 15, 30, 50, 70]})
    joined = kw.pop("joined", False)
    cs = None

    if joined:
        cs = pickle_load('collocs.pkl')
        
    tk = build_joined_tokenizer(count=True, ngrams=cs['bgs'], ngram_range=(1,2)) \
        if joined else build_tokenizer(count=True, ngram_range=ngrams)
    print("Tokenizing...")
    toks = tk.fit_transform(data)
    return lda_optimize_model(toks, grid, **kw)
    
    
# Higher log-likelihood and lower perplexity (exp(-1 * log-likelihood / word))
# is better
def print_lda_optim_results(model, kind='gensim', data=None, **kw):
    """Print best model results from model optimization."""
    sections = kw.pop("sections", False)
    scorer = 'Coherence' if model.scorer_.__name__ is 'lda_gensim_score' \
        else 'log-likelihood'
    estimator_scorer = model.best_estimator_.score
    perp_func = model.best_estimator_.score if kind is 'gensim' else \
        model.best_estimator_.perplexity
    print("LDA optimized using grid search", end='')
    if sections:
        print(" on sections '" + ' '.join(sections) + "'", end='')
    print()
    print(f"The scoring function used here is {scorer}")
    print("--------------------")
    print(model)
    print()
    print("Best LDA model\n--------------------\n")
    print("  - Params:         ", model.best_params_)
    print(f"  - {scorer}:      ", model.best_score_)
    if data is not None:
        print("  - Perplexity:     ", perp_func(data))
    print()
    print("Best Model:")
    print(model.best_estimator_)
    

def plot_lda_optim_topics(model, kind='gensim', **kw):
    """Plot optimized model scoring-function results vs. number of topis."""
    sections = kw.pop('sections', False)
    title = kw.pop('title', "Optimal Number of Topics for LDA") 
    if sections:
        title += '\ntrained on ['
        title += (', '.join(sections) if isinstance(sections, list) \
            else sections) + ']'
    score_func = 'Coherence Score' if model.scorer_.__name__ is 'lda_gensim_score' \
        else 'Log-Likelihood'
    pgrid_key = 'num_topics' if kind == 'gensim' else 'n_components'
    res = model.cv_results_
    lls = [round(s, 2) for s in res['mean_test_score']]
    sns.lineplot(x=model.param_grid[pgrid_key], y=lls, **kw)
    plt.title(title)
    plt.xlabel("Number of Topics")
    plt.ylabel(f"{score_func}")
    plt.show()
