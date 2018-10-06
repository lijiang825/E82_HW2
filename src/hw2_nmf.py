#!/usr/bin/env python

"""
Run NMF model
"""

from sklearn.decomposition import NMF

from hw2_config import *
from hw2_tokenize import build_tokenizer
from hw2_data import NipsData
from hw2_model_analysis import print_top_words

nips = NipsData()
dat = nips.load_data(sample_frac=0.4)


# NMF params
n_topics = 20
n_top_words = 10

# Use TF-IDF with NMF

tk = build_tokenizer()
tfidf = tk.fit_transform(dat.abstract)
pp.pprint(tk.get_feature_names()[:10])

# Run NMF model
nmf = NMF(n_components=n_topics, random_state=RANDOM_SEED, alpha=0.1, l1_ratio=0.5)
nmf.fit(vect)

# Topics
print_top_words(nmf, tk.get_feature_names(), n_top_words,
                title=f"Top {n_top_words} found with NMF:")


# Find closest papers to each topic
nmf_embedding = nmf.transform(tfidf)
top_idx = np.argsort(nmf_embedding, axis=0)[-:3]
