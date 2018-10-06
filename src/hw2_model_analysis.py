#!/usr/bin/env python

"""
Analyze topic models.
"""

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as patches
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from matplotlib.colors import LogNorm

from hw2_config import *
from hw2_lda_gensim import *
from hw2_tokenize import *
from hw2_tokenizers import *
from hw2_lda import *
from sklearn.linear_model import LinearRegression


## -------------------------------------------------------------------
### Plotting configs

cols = {
    "dark2" : ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e',
             '#e6ab02','#a6761d','#666666'],
}
dark2_cmap : ListedColormap(cols['dark2'])

# mpl.style.use('seaborn-dark-palette')
# current_palette = sns.color_palette()
    
# set2_colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f',
#                '#e5c494','#b3b3b3']
# light_grey = np.array([float(248)/float(255)]*3)
# shade_black = '#262626'

def set_mpl_params(**kw):
    """Set matplotlib params."""
    font = kw.pop('font', { 'family': 'monospace', # 'StixGeneral'
                            'weigth': 'bold',
                            'size': 12 })       # 'larger'
    fig = kw.pop('fig', { 'figsize' : (10, 6),
                          'dip' : 100 })
    lines = kw.pop('lines', { 'linewidth' : 2 })
    patch = kw.pop('patch', { 'edgecolor' : 'white',
                              'facecolor' : cols['dark2'][0] })
    axes = kw.pop('axes', { 'facecolor' : 'white' })
    rcParams('font', **font)
    rcParams('figure', **fig)
    rcParams('lines', **lines)
    rcParams('axes', **axes)
    rcParams('axes.prop_cycle').by_key()['color'][1]
    
    # set back to defaults: plt.style.use('default')
    style = kw.pop('style', 'ggplot')
    plt.style.use(style)


def cible_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Make a target axis at 0,0 with ticks along the axis lines.
    
    The top/right/left/bottom keywords toggle whether the corresponding plot
    border is drawn.
    """
    ax = axes or plt.gca()

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

## -------------------------------------------------------------------
### Generic

def plot_word_freqs(words, counts, n_words=10):
    """Words frequences as barplot."""
    fig, ax = plt.subplots(1, 1)
    rng = np.arange(n_words)[::-1]
    sns.barplot(rng, counts[-n_words - 1:-1], palette='hls', ax=ax)
    ax.set_xticks(rng)
    ax.set_xticklabels(words[-n_words - 1:-1], rotation=45)
    plt.title("Word Frequencies")
    plt.show()


## -------------------------------------------------------------------
### Scikit-learn models

def print_lda_model_diagnostics(model, data, kind='gensim'):
    """Print model diagnostics: log-likelihood and perplexity."""
    print(model)
    print("Params:")
    pp.pprint(model.get_params())
    print("--------------------")
    print("Performance:")
    print("  - Log Likelihood: ", model.score(data))
    print("  - Perplexity:     ", model.perplexity(data))


def print_top_words(model, feature_names, n_top_words, **kw):
    """Print top weighted words in model -- see scikit-learn tutorial."""
    title = kw.pop("title", f"Top {n_top_words} for each topic")
    print(title)
    print()
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ' '.join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
        
def plot_lda_topic_barplot(model, topic, feature_names, n_words, **kw):
    """Plot barplot of weights of top n_words in topic."""
    fig, ax = plt.subplots(1, 1)
    inds = model.components_[topic].argsort()[:-n_words - 1:-1]
    sns.barplot(x=np.array(feature_names)[inds], y=model.components_[topic][inds],
                palette='hls', ax=ax) # palette='Blues_d'
    ax.set_xticks(np.arange(n_words))
    ax.set_xticklabels(np.array(feature_names)[inds], rotation=45)
    fig.subplots_adjust(bottom=0.2)
    ax.yaxis.grid()
    plt.title(f"Topic #{topic}: top {n_words} words")
    plt.show()


def print_closest_matches(model, vect, df, num_titles=3, **kw):
    """Print the paper section (default title) of the closest
    matching papers to each topic."""
    section = kw.pop("section", 'title')
    embeddings = model.transform(vect)
    vecs = np.argsort(embeddings, axis=0)[-num_titles:]

    i = 0
    for inds in vecs.T:
        print(f"Topic {i}:")
        for ind in inds:
            print(f"\t {df.iloc[ind][section]:>10}")
        i += 1
        
def tsne_projection(data, random_state=RANDOM_SEED, **kw):
    """Plot TSNE projection of data."""
    mod = TSNE(random_state=random_state, **kw)
    proj = mod.fit_transform(data)
    dat = pd.DataFrame(proj, columns=['x', 'y'])
    dat['hue'] = proj.argmax(axis=1) # coloring for classes
    return dat


def plot_projected(proj, data, colors=cols['dark2']):
    col_vals = {c : cols['dark2'][i] for i, c in enumerate(data["class"].unique())}
    cols = [col_vals[c] for c in data["class"]]
    ax = plt.gca()
    ax.scatter(proj[0], proj[1], c=cols)
    ax.grid(True)
    ax.legend(title="Class", loc="best", labels=col_vals.keys(),
              handles=[Patch(facecolor=col_vals[i]) for i in col_vals])

## -------------------------------------------------------------------
### LDA w/ gensim

nips = NipsData()
nips.load_data(sample_frac=1)

sections = ['title', 'abstract']
dat = nips.combined_sections(sections, data=nips.raw)
sname = get_save_name(sections)
yrs = nips.raw.year

# used by lda_gensim_score
docs = lda_get_corpus(dat, save=True, name=sname)
d, bow = lda_get_dictionary(dat, save=True, name=sname)

# used by sci lda
cs = pickle_load('collocs.pkl')
tk = build_joined_tokenizer(count=True, ngrams=cs['bgs'])
toks = tk.fit_transform(dat)

lda = pickle_load('lda_sci_20_title_abstract')
# inds = np.argmax(trans, axis=1) # index of dominant topic

def topics_sci_top_words(model, topics, feature_names, n_top_words):
    """Top n words for model topics."""
    tops = {}
    for t in topics:
        tops["Topic"+str(t)] = [feature_names[i] for i in
                              model.components_[t].argsort()[:-n_top_words:-1]]
    return pd.DataFrame(tops)


def topics_sci_weight_by_year(model, vect, yrs):
    """Get normalized weights of topics / doc / year"""
    trans = model.transform(vect)   # normalized by topic
    df = pd.DataFrame(trans)
    out = df.groupby(yrs).apply(sum)
    out = pd.DataFrame(out.values / out.values.sum(axis=1)[:, np.newaxis],
                       columns=["T" + str(i) for i in range(model.n_components)])
    out['year'] = yrs.value_counts(ascending=True).index
    return out


def rank_topics_by_slope(topics):
    """Rank topics by slopes over years."""
    lm = LinearRegression()
    xs = np.arange(topics.shape[0]).reshape(-1,1)
    slopes = topics.iloc[:,:-1].apply(lambda ys: lm.fit(xs, ys.values).coef_)
    inds = np.argsort(-slopes.values)[0]
    df = topics.iloc[:, :-1]
    df = df.iloc[:, inds]
    df['year'] = topics['year']

    return df, inds


def plot_topics_by_year(topics, n_topics):
    """Plot topics matrix as lineplot over the years.
    Expects a dataframe with a year and topics columns."""
    dat, inds = rank_topics_by_slope(topics)
    dat = dat.iloc[:, :n_topics]
    dat['year'] = topics['year']
    dat = dat.melt(id_vars=['year'], var_name='topic', value_name='weight')
    
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x='year', y='weight', hue='topic', data=dat)
    plt.title("Top trending topics from NIPs")
    plt.xticks(range(1987, 2018, 5))
    plt.show()
    return inds


def topic_doc_weights_by_year(model, bow, yrs):
    """
    Determine weight of each topic for each year. This the sum of a
    topics contribution to all documents in the year. Yearly topic distributions 
    are normalized to sum to 1.
    """
    doc_dist = model.get_document_topics(bow) # topic contributions to each document
    idx = yrs - min(yrs)
    mat = np.zeros((len(np.unique(yrs)), model.num_topics))
    
    # compute topics contribution to all documents in year
    for i, dist in enumerate(doc_dist):
        for topic, perc in dist:
            mat[idx[i], topic] += perc

    # normalize yearly topic sums
    res = pd.DataFrame(
        mat / mat.sum(axis=1)[:, np.newaxis],
        columns=[f'Topic {i}' for i in range(model.num_topics)])
    res['year'] = yrs.value_counts(ascending=True).index
    return res


def topic_documents_by_year(model, bow, yrs, perc=False):
    """
    Determine the percentage of documents each year where a topic
    was the main contributor. Results are ordered by topic coherence score
    given the text passed.

    If perc=True, then the weight given the topic corresponds to its
    actual probability distribution in the document. Otherwise,
    A dominant topic is assumed to own the whole document.
    """
    doc_dist = model.get_document_topics(bow) # topic contributions to each document
    idx = yrs - min(yrs)
    mat = np.zeros((len(np.unique(yrs)), model.num_topics))
    
    # number of documents where topic is the main contributor
    for i, dist in enumerate(doc_dist):
        mat[idx[i], dist[0]] += dist[0][1] if perc else 1

    # normalize yearly topic sums
    res = pd.DataFrame(
        mat / mat.sum(axis=1)[:, np.newaxis],
        columns=[f'Topic {i}' for i in range(model.num_topics)])
    
    res['year'] = yrs.value_counts(ascending=True).index
    return res
    
def print_gensim_top_topics(model, topics):
    """Print the top words for each topic and their weights"""
    for t in topics:
        print(model.show_topic(t))
