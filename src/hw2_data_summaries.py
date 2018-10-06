#!/usr/bin/env python

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from hw2_config import *

## -------------------------------------------------------------------
### Simple Exploratory Stuff

def data_info(data):
    """Print basic data info."""
    print(f"Columns: {data.columns}")
    print(f"Shape: {data.shape}")
    print(f"Data:\n{data.head(5)}")
    print(f"...")

# Papers over time: generally exponentially growing #papers/year
def data_papers_per_year(data):
    """Barplot of NIPs papers per year."""
    sns.countplot(y='year', data=data)
    plt.grid(axis='x')
    plt.title("NIPS Articles per Year")
    plt.xlabel("Articles")
    plt.ylabel("Year")
    plt.show()

# Papers w/ References
def data_refs_info(data):
    """Show info on extracted references from articles."""
    if "refs" in data:
        print(f"Extracted references from {data['refs'].count()} texts")
        refs = data['refs'].dropna().apply(len)
        sns.countplot(x=refs)
        plt.title("Extracted Reference Counts / Article")
        plt.ylabel("Number of NIPs Articles")
        plt.xlabel("Reference Counts")
        ticks = np.unique(refs)
        plt.xticks(ticks[::5], np.arange(1, max(refs))[ticks[::5]])
        plt.show()


# Word distributions
def data_word_distributions(data):
    """Rough distributions of word counts in titles, article, and texts."""
    sns.set(style="white", color_codes=True) # minimal
    fig, axs = plt.subplots(ncols=3, facecolor='w', edgecolor='k')
    
    for i, col in enumerate(["title", "abstract", "text"]):
        sns.distplot(data[col].str.split('\s*').map(len), ax=axs[i])
        axs[i].set_xlabel(col)
        axs[i].set_yticklabels([])

    fig.suptitle("Distributions of word counts in NIPs corpus")
    plt.show()
