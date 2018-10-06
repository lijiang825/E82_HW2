#!/usr/bin/env python

"""
Run DTM model -- requires eternally installing.
"""
import os
import numpy as np
import pandas as pd

from hw2_config import *
from hw2_lda_gensim import *
from hw2_data import NipsData

from gensim.models.wrappers import DtmModel

dtm_path=os.path.join(root_path(), "../dtm/dtm/main")
dtm_defaults = dict(
    rng_seed=RANDOM_SEED,
    num_topics=15
)

@timeit
def dtm_run(data, times, dtm_path, **kw):
    """Run DTM model."""
    sname = kw.pop("name", '_temp_')
    save = kw.pop('save', True)
    d, bow = lda_get_dictionary(data, save=save, name=sname)

    key = f"lda_dtm_{ncomps}_" + sname
    if os.path.exists(os.path.join(PKLDIR, key)):
        return pickle_load(key)
    else:
        mod = DtmModel(dtm_path=dtm_path, corpus=bow, id2word=d,
                    time_slices=times, **dtm_defaults)
        mod.save(os.path.join(PKLDIR, key))
        return mod

# DTM Analysis
nips = NipsData()
dat = nips.load_data(sample_frac=1)
sections = ['title', 'abstract']
ncomps = dtm_defaults['num_topics']
dat = nips.combined_sections(sections, data=dat)
yrs, cnts = papers_per_year(nips.raw)
sname = get_save_name(sections)
docs = lda_get_corpus(dat, name=sname, save=True)
d, bow = lda_get_dictionary(dat, name=sname, save=True)
## mod = dtm_run(dat, cnts, dtm_path, name=sname, save=True)
mod = DtmModel.load(os.path.join(PKLDIR, 'lda_dtm_15_title_abstract'))


def print_dtm_top_words_for_year(model, years, n_topics, n_words):
    """Print top n_words from top n_topics for year in years."""
    print(f"Top {n_words} from top {n_topics} for year(s) {years}:")
    yrs = enumerate(range(1987, 2018))
    inds = [(i,yr) for i, yr in yrs if yr in years]
    
    for i, yr in inds:
        print(f"Year {yr}:")
        for topic, words in enumerate(model.dtm_coherence(i, n_words)[:n_topics]):
            print(f"  Topic #{topic}: " + ', '.join(words))

def dtm_coherence(model, corpus, d, year):
    """Get coherence for DTM model at year."""
    years = enumerate(range(1987, 2018))
    ind = [i for i,yr in years if yr == year][0]
    tw = model.dtm_coherence(time = ind)

    cm = CoherenceModel(topics=tw, texts=corpus, dictionary=d, coherence='c_v')
    return cm.get_coherence()

def print_dtm_coherence_by_year(model, corpus, d):
    """Print Coherence scores for each year in DTM model."""
    years = enumerate(range(1987, 2018))

    print("Coherence scores for each year:")
    for i, yr in years:
        coh = dtm_coherence(model, corpus, d, yr)
        print(f"{yr}: {coh:0.4f}")


def dtm_topics_by_year(model, corpus, yrs):
    """Calculate the normalized weights of topics over time and the most
    relevant words."""
    topics = model.num_topics
    num_yrs = len(yrs.unique())
    mat = np.zeros((num_yrs, model.num_topics))
    words = pd.DataFrame()
    
    for yr in range(num_yrs):
        for t in range(topics):
            dist = model.show_topic(t, yr)
            ws = []
            for val, word in model.show_topic(t, yr):
                mat[yr, t] += val
                ws.append(word)
            words[f'Topic {t}'] = ws

    res = pd.DataFrame(mat / mat.sum(axis=1)[:, np.newaxis],
                       columns=[f'Topic {i}' for i in range(model.num_topics)])
    res['year'] = yrs.value_counts(ascending=True).index
    return res, words


def plot_dtm_topic_term_distr(model, corpus, year, topic):
    doc_topic, topic_term, doc_lengths, term_frequency, vocab = \
        mod.dtm_vis(time=i, corpus=bow)
    fig, ax = plt.subplots(1, 1)
    sns.distplot(topic_term[topic])
    plt.show()
