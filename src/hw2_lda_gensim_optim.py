#!/usr/bin/env python

from hw2_config import *
from hw2_lda_gensim import *
from hw2_data import NipsData
from hw2_lda import plot_lda_optim_topics, print_lda_optim_results

nips = NipsData()
dat = nips.load_data(sample_frac=1)
sections = ['title', 'abstract', 'text']
sname = get_save_name(sections)
key = 'lda_gensim_' + sname
dat = nips.combined_sections(sections, data=dat)

# used by lda_gensim_score
docs = lda_get_corpus(dat, save=True, name=sname)
d, bow = lda_get_dictionary(dat, save=True, name=sname)

def lda_gensim_score(estimator, X, y=None):
    """To pass to grid search. Note: docs must be defined globally, this
    should be wrapped in optimize, but can't pickle in that case."""
    cm = CoherenceModel(model=estimator.gensim_model, texts=docs,
                        dictionary=estimator.gensim_model.id2word,
                        coherence='c_v')
    return cm.get_coherence()

@timeit
def lda_gensim_optimize(data, pgrid={'num_topics': [10, 15, 20, 30]},
                        **kwn):
    """Optimize number of topics using coherence score."""
    d, bow = lda_get_dictionary(data)

    obj = LdaTransformer(id2word=d, num_topics=10, iterations=50, passes=8,
                         alpha='symmetric', chunksize=4000)
    mod = GridSearchCV(obj, param_grid=pgrid, scoring=lda_gensim_score)
    mod.fit(bow)

    return mod

optims = pickle_load('lda_optim.pkl') or dict()
if key not in optims:
    mod = lda_gensim_optimize(dat)
    optims[key] = mod
    pickle_save(optims, 'lda_optim.pkl')
