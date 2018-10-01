
# Table of Contents

1.  [Notes](#org15290a4)

My list of notes on looking over the code. I like the method you used, it's
solid and easy to understand. What do you think about doing something similar
with (multiple regression) with the weighted vectors returned by LDA or another
method?


<a id="org15290a4"></a>

# Notes

-   I added a fair amount of data cleanup in the hw2<sub>data.py</sub> module. It extracts
    the "References" sections from the texts, parses them into lists, and removes
    them from the texts. This is all configurable, but by default it will add
    another column, "refs", to the dataset. I thought it might be useful if there
    was time as an additional linkage between topics. However, not all texts have
    references and they are in different formats making them hard to parse.

-   It would be nice to have access to the same dataset; I've been using the cleaned
    version that someone posted on [piazza](https://drive.google.com/file/d/180FBOXqxdyvzHihHsg_bAbYS-UF1WrKZ/view?usp=sharing). It appears to have a few more stuff
    from the early years, eg. 1987 than yours. It would be nice to have a way to
    access it directly from python, but I haven't used the google drive API,
    although I saw there is one for Python.

-   I'm using a modified word tokenizer with added lemmatization which seems to
    work faily well (using nltk and wordnet) along with the Tfidfvectorizer and
    CountVectorizer. As a preprocessing step, these vectorizers already do things
    like strip punctuation, convert to lowercase, fixup unicode, etc.

-   I'm using wordnet to do lemmatization during tokenization. If you have **make**
    on your computer you should be able to just run **make nltk-data** and it will
    download the lemma corpus in this directory. Otherwise, it is simple to
    install with nltk.download.

-   I've been thinking a bit about how to incorporate time into
    the LDA model. There are various techniques out there, but they may be more
    work than we want to put in. I put some of the refs I was looking at in
    ./etc folder along with some notes and other citations. You don't have to look
    at this at all if you don't want to, I can focus on doing LDA if you like.
    
    I was thinking that even without have year as a covariate, an LDA model would
    still be useful to run on the whole dataset. Then, the probability of a
    document belonging to a topic can still be plotted over time.

-   Skewed data - the number of documents grows exponentially over time. So, I
    think we account for that when running models. It doesn't seem like it is
    important for the relative freq. you've done, but I'm thinking about LDA where
    the sample size is so much greater as years increase, so the topics will be
    skewed towards those present more recently.

-   I'll convert all my code to a notebook eventually, I just hate programming
    in the notebooks.

