
My list of notes on looking over the code. I like the method you used, it's
solid and easy to understand. What do you think about doing something similar
with (multiple regression) with the weighted vectors returned by LDA or another
method?

# Notes

-   [ ] we need to have access to the same dataset; I've been using the cleaned
    version that someone posted on [piazza](https://drive.google.com/file/d/180FBOXqxdyvzHihHsg_bAbYS-UF1WrKZ/view?usp=sharing). It would be nice to have a way to
    access it directly from python, but I haven't used the google drive API,
    although I saw there is one for Python. When I create tokens, I'm not getting
    quite the same results as you, and I was thinking it may have to do with the
    dataset.

-   [ ] I think the tokenization can be simplified, the
    TfidfVectorizer/CountVectorizer can do things like convert to lowercase,
    remove stopwords, normalization, etc. It will also generate stopwords on the
    fly from words that occur very frequently (using the max<sub>df</sub> param) as far as I
    can tell.

-   [ ] I modified the word tokenizer slightly since there were lots of numbers
    turning into tokens.

-   [ ] The lemmatization seems pretty good - it might be worth trying out. I was
    using the wordnet one. If you have **make** on your computer you should be able
    to just run **make nltk-data** and it will download the lemma corpus in this
    directory.

-   [ ] LDA model - I've been thinking a bit about how to incorporate time into
    the LDA model. There are various techniques out there, but they may be more
    work than we want to put in. I put some of the refs I was looking at in
    ./etc folder along with some notes and other citations. You don't have to look
    at this at all if you don't want to, I can focus on doing LDA if you like.
    
    I was thinking that even without have year as a covariate, an LDA model would
    still be useful to run on the whole dataset. Then, the probability of a
    document belonging to a topic can still be plotted over time.

-   [ ] Skewed data - the number of documents grows exponentially over time. So, I
    think we account for that when running models. It doesn't seem like it is
    important for the relative freq. you've done, but I'm thinking about LDA where
    the sample size is so much greater as years increase, so the topics will be
    skewed towards those present more recently.

-   [ ] I'll convert all my code to a notebook eventually, I just hate programming
    in the notebooks.

