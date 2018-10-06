#!/usr/bin/env python

import pandas as pd
import os

from gensim.models import LdaSeqModel, LdaModel, LdaMulticore, Phrases, CoherenceModel
from gensim import corpora, models
from gensim.corpora import Dictionary

# integration with sklearn
from gensim.sklearn_api import LdaTransformer, LdaSeqTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from hw2_config import *
from hw2_tokenize import tokenize_lemmatize
from hw2_data import NipsData

# default LDA params
lda_gensim_defaults = dict(
    num_topics = 18,
    decay = 0.5,
    passes = 30,
    random_state=RANDOM_SEED,
    alpha='auto',
    eta='auto',
    per_word_topics=True,
    eval_every=None,
    chunksize=4000,
    iterations=400
)

# time_slice: list of ints, eg. counts / year
# take a loooong time
lda_seq_defaults = dict(
    num_topics = 15,
    passes = 5,
    random_state = RANDOM_SEED,
    initialize = 'gensim',      # initialize with gensim model unless have one
    chunksize = 4000,
    em_max_iter = 5
)

## -------------------------------------------------------------------
### Prep

def _docs_add_ngrams(docs, grams):
    """Add ngrams to docs, keeping orignial unigrams. Otherwise, when reducing
    the words by frequency of occurence, common unigrams can be kept."""
    for idx in range(len(docs)):
        for token in grams[docs[idx]]:
            if '_' in token:
                docs[idx].append(token)
    return docs


def docs_add_ngrams(docs, trigram=False, **kw):
    """Add ngrams to docs. If trigram is True, add both bi/trigrams.
    Optional arguments:
      bg_min_count: passed to bigram Phrases
      bg_threshold: passed to bigram Phrases
    All others are passed to trigram Phrases.
    """
    bg_min_count = kw.pop("bg_min_count", 5)
    bg_threshold = kw.pop("bg_threshold", 100)
    scoring = kw.pop("scoring", 'default')
    ds = list(docs.values)

    bigrams = Phrases(ds, min_count=bg_min_count, threshold=bg_threshold,
                      scoring=scoring)
    ds = _docs_add_ngrams(ds, bigrams)

    if trigram:
        trigrams = Phrases(bigrams[ds], scoring=scoring, **kw)
        ds = _docs_add_ngrams(ds, trigrams)

    return pd.Series(ds, index=docs.index)


def docs_to_dict(docs, **kw):
    """Convert docs to Dictionary and BOW, filtering common/rare words.
    Returns (dictionary, BOW)"""
    no_below = kw.pop("no_below", .02)
    no_above = kw.pop("no_above", 0.9)
    d = Dictionary(docs)
    d.filter_extremes(no_below=no_below, no_above=no_above, **kw)
    d.compactify()
    return d, docs.apply(d.doc2bow)


def lda_get_corpus(data, **kw):
    """
    Generate corpus from raw data. This involves all of the preprocessing
    steps.

    Optional args:
      - name: save filename
      - save: True if should save corpus (if no name, uses 'temp_corpus')
    """
    keep_joins = kw.pop("keep_joins", False)
    use_trigrams = kw.pop("use_trigrams", True)
    name = kw.pop("name", None)
    save = kw.pop("save", False)
    savepath, exists = get_save_path(name if name else "temp_corpus") if save\
        else (None, False)

    if name and exists:
        return pickle_load(savepath)
        
    print("Tokenizing...")
    docs = tokenize_lemmatize(data, keep_joins=keep_joins)
    print("Adding N-grams...")
    docs = docs_add_ngrams(docs, trigram=use_trigrams)
    if save and name:
        docs.to_pickle(savepath)

    return docs


def lda_get_dictionary(data, **kw):
    """Create gensim BOW/Dictionary from raw data using defaults."""
    no_below = kw.pop("no_below", 0.02)
    no_above = kw.pop("no_above", 0.6)
    name = kw.get('name', None)  # prefix for saves, passs on
    save = kw.get('save', False)
    savepath = get_save_path(name)[0] if name else None
    
    if name and os.path.exists(savepath + "_bow"):
        bow = pickle_load(savepath + "_bow")
        d = corpora.Dictionary.load(savepath + "_dict")
    else:
        docs = lda_get_corpus(data, **kw)
        d, bow = docs_to_dict(docs, no_below=no_below, no_above=no_above)

        if save:                    # save dictionary, and BOW
            d.save(savepath + "_dict")
            bow.to_pickle(savepath + "_bow")

    return d, bow


def lda_data2bow(data, **kw):
    """Convert data to tokens, BOW, and dictionary."""
    docs = lda_get_corpus(data, **kw)
    return docs, lda_get_dictionary(docs, **kw)
    
## -------------------------------------------------------------------
### Run Models

def lda_load_model(fname):
    """Try to load a model if already trained."""
    pklpath, exists = get_save_path(fname)
    return LdaModel.load(pklpath) if exists else None

    
@timeit
def run_lda_gensim(data, **kw):
    """
    Constructs gensim BOW from data, adding bigrams and trigrams by default.
    Uses lda_gensim_defualts, but they are overriden by passed params.
    """
    kw.pop('name')
    d, bow = lda_get_dictionary(data, **kw)
    lda_args = {**lda_gensim_defaults, **kw}
    lda_args.pop('name', None)
    lda_args.pop('save', None)
    return LdaModel(corpus=bow, id2word=d, **lda_args)
    

@timeit
def run_lda_gensim_all_models(nips, params=lda_gensim_defaults,
                              sections=[["title", "abstract"]], **kw):
    """Run LDA on sections of NIPs dataset."""
    for section in sections:
        dat = nips.raw[section] if isinstance(section, str) else \
            nips.raw[section].apply(lambda x: '\n'.join(x), axis=1)
        name = get_save_name(section)
        ntopics = kw.get('num_topics', params['num_topics'])
        modname = f"lda_{ntopics}_" + name

        mod = lda_load_model(modname)
        if mod:
            print(f"Loaded saved LDA model: {modname}")
        else:
            print(f"Running LDA on {section}...")
            mod = run_lda_gensim(dat, name=name, save=True, num_topics=ntopics)
            path, have = get_save_path(modname)
            mod.save(path)

## -------------------------------------------------------------------
### Time-varying LDA

def papers_per_year(data):
    """Counts of papers published / year."""
    cnts = data.year.value_counts()
    return list(cnts.index), list(cnts.values)


def get_nips_combined(sections, data=None):
    """Combined NIPs sections."""
    nips = NipsData()
    nips.load_data()
    return nips.combined_sections(sections, data)


# takes forever, hours / day
@timeit
def run_ldaseq(data, sections, **kw):
    """Run LDA sequential model."""
    nips = NipsData()
    nips.load_data()
    yrs, cnts = papers_per_year(data)
    data = nips.combined_sections(sections, data)
    name = get_save_name(sections)
    ncomps = kw.get('num_topics', lda_seq_defaults['num_topics'])
    modname = f"ldaseq_{ncomps}_" + name

    mod = lda_load_model(modname)
    if mod:
        return mod

    d, bow = lda_get_dictionary(data, name=name, save=True)

    print(f"Running LDAseq on {sections}")
    lda_args = {**lda_seq_defaults, **kw}
    mod = LdaSeqModel(corpus=bow, time_slice=cnts, id2word=d, **lda_args)

    path, _ = get_save_path(modname)
    mod.save(path)
    return mod


def lda_gensim_to_sci(data, sections, n_topics, **kw):
    """Wrap gensim LDA model for scikit-learn."""
    dat = get_nips_combined(sections, data)
    d, bow = lda_get_dictionary(d, **kw)
    
    args = {**lda_gensim_defaults, **kw}
    args.pop('per_word_topics')
    args['num_topics'] = n_topics
    return LdaTransformer(id2word=d, **args)


## -------------------------------------------------------------------
### Model metrics

def lda_coherence(model, corpus, dic):
    """Compute the coherence score."""
    cm = CoherenceModel(model=model, texts=corpus, dictionary=dic, coherence='c_v')
    return cm.get_coherence()
