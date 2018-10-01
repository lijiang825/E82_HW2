#!/usr/bin/env python

"""
Download prereqs for NLTK lemmatization/tokenization into root directory.
"""
import nltk
from hw2_utils import *
from hw2_config import *

def main():
    nltk.download(['wordnet', 'punkt'], download_dir=root_path())

if __name__ == '__main__':
    main()
