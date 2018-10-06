#!/usr/bin/env python

"""
Various classes of tokenizers
"""

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import string
import regex as re
import unicodedata
import sys
from nltk.stem.porter import PorterStemmer

def text_preprocess(doc, keep_joins=False):
    """
    Simple text prepocessing:
      - remove non-ascii and punctuation. Leaves [._-] unless keep_joins is False.
      - lowercase
    """
    nonascii = '[^\x00-\x7F]+'
    punkt = '[^.0-9A-Za-z_ \t\n\r\v\f-]'
    no_joins = '[^.0-9A-Za-z \t\n\r\v\f]'
    re_joins = re.compile(nonascii + "|" + punkt)
    re_nojoins = re.compile(nonascii + "|" + no_joins)

    return re.sub(re_joins if keep_joins else re_nojoins, ' ', doc).lower()


# Lemmatizer: can be used as a replacement tokenizer + lemmatizer in
# scikit-learn tokenizers
class LemmaTokenizer(object):
    """
    Regexp tokenizer (words >= 3) and lemmatizer. Can be passed to 
    scikit-learn vectorizers. By default, lemmatizes nouns, adjectives, 
    adverbs, and verbs. Converts '-' to '_' and leaves them in words if
    keep_joins is True.

    Optional arguments:
      - pos: one of, or list of, ['N', 'V', 'R', 'N'] for noun, verb, adverb, verb
        these determine which words get kept and lemmatized.
    """
    tags = {'J': wn.ADJ, 'V': wn.VERB, 'R': wn.ADV, 'N': wn.NOUN}
    
    def __init__(self, keep_joins=False, **kw):
        self.keep_joins = keep_joins
        self.wnl = WordNetLemmatizer()
        self.rt = RegexpTokenizer('(?ui)\\b[A-Za-z][A-Za-z0-9_-]*\\b')\
            if keep_joins else RegexpTokenizer('(?ui)\\b[A-Za-z][A-Za-z0-9]*\\b')
        self.pos = kw.pop('pos', False)
        if self.pos:
            self.tags = {i: self.tags[i] for i in self.pos}
        
    def __call__(self, doc):
        return list(filter(lambda x: len(x) > 2,
                           [self.wnl.lemmatize(w, pos=self.tags[t[0]])\
                            # .replace('-', '_')
                            for w, t in pos_tag(self.rt.tokenize(doc))
                            if t[0] in self.tags]))

    def __str__(self):
        return f"{type(self).__name__}(lemmatizer={self.wnl},\n\tpos={self.tags},\n\t\
tokenizer={self.rt})"
        
    def __repr__(self):
        return self.__str__()


class GensimTokenizer(LemmaTokenizer):
    """
    Remove stopwords and use nltk's word tokenizer instead of regex.
    Optionally, stem words and convert '-' to '_', used when creating ngrams.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.sw = set(stopwords.words('english'))
        self.stem = kw.pop("stem", False)
        self.ps = PorterStemmer()
        self.translate = kw.pop("translate", False)

    def __call__(self, doc):
        res = [self.wnl.lemmatize(w, pos=self.tags[t[0]]) for w,t in
               pos_tag(word_tokenize(doc)) if t[0] in self.tags and len(w) > 2]
        stem = [self.ps.stem(w) for w in res] if self.stem else res
        return list(filter
                    (lambda x: len(x) > 2 and x[0] not in ['-', '_'],
                     [w.replace('-', '_') for w in stem]))\
                     if self.translate else stem


class NgramPreTokenizer(LemmaTokenizer):
    """
    Like NgramLemmaTokenizer but also does preprocessing:
      - converts to lowercase
      - removes unicode / punctuation
      - removes english stopwords
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.tbl = dict.fromkeys( # translation table to remove unicode/punct
            i for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith('P'))
        self.sw = stopwords.words('english')
        
    def __call__(self, doc):
        return list(filter(lambda x: len(x) > 2,
                           [self.wnl.lemmatize(w.lower().translate(self.tbl),
                                               pos=self.tags[t[0]])
                            for w,t in pos_tag(self.rt.tokenize(doc))
                            if t[0] in self.tags and w not in self.sw]))


class NgramLemmaTokenizer(LemmaTokenizer):
    """
    Works like the LemmaTokenizer, but replaces collocated ngrams with joined
    versions, eg. 'neural network' => 'neural_network'.
    
    This attempts to address the BOW assumption that word locations are
    independent.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.ngrams = kw.pop('ngrams', None)
        
    def __call__(self, doc):
        res = super().__call__(doc)
        return self.append_ngrams(res, self.ngrams) if self.ngrams is not None\
            else res

    @staticmethod
    def append_ngrams(lst, ngrams):
        """
        Append ngrams to lists with their joined forms, eg. 
        ['neural', 'neural', 'network'] => ['neural', 'neural_network'].
        Note: leaves unigrams in place so they can later be removed by 
        frequency trimming.
        """
        n = len(ngrams[0]) - 1
        for i in range(n, len(lst)):
            if tuple(lst[(i-n):(i+1)]) in ngrams:
                lst.append('_'.join(lst[(i-n):(i+1)]))
        return lst

    @staticmethod
    def replace_ngrams(lst, ngrams):
        """
        Replace ngrams with their joined forms, eg. 
        ['neural', 'neural', 'network'] => ['neural', 'neural_network'].
        """
        n = len(ngrams[0]) - 1
        res = lst[:(n-1)]
        skips = 0
        for i in range(n, len(lst)):
            if skips > 0:
                skips -= 1
                continue
            if tuple(lst[(i-n):(i+1)]) in ngrams:
                res.append('_'.join(lst[(i-n):(i+1)]))
                skips += 1
            else:
                res.append(lst[i-n])
        if n - skips > 0:       # add tailing tokens if not merged
            res += lst[-n+skips:]
        return res
