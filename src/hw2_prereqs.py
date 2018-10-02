#!/usr/bin/env python

"""
Download prereqs for NLTK lemmatization/tokenization into root directory.
"""
from hw2_config import *
from hw2_tokenize import get_nltk_prereqs

def main():
    get_nltk_prereqs()

if __name__ == '__main__':
    main()
