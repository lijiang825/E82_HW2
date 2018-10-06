SHELL       = /bin/bash
conda       ?= conda
python      ?= python

SETUP       = setup
DATADIR     = data
ETCDIR      = etc
SRCDIR      = src
NLTK_DATA   = corpora tokenizers
GITDIR      = $(realpath E82_HW2)

DATACOLS    = title,abstract
SAMPLE      = 0.8
DRIVER      = ${SRCDIR}/hw2_cli --cols ${DATACOLS} --sample ${SAMPLE}

# just for converting from .org to markdown or whatever format
emacs       ?= emacs
EMACS_FLAGS ?= -batch -Q
EMACS_MD    = ${EMACS_FLAGS} -f org-md-export-to-markdown

all:

# setup python env called 'text' with conda if installed
create-env:
	@if hash conda 2>/dev/null && ! conda env list | grep "envs/text"; \
	then                                                               \
		conda env create -f ${SETUP}/text.yml;                     \
	fi

# download nltk lemma data
.PHONY: nltk-data pickle
nltk-data:
	@(if hash conda; then source activate text; fi;                    \
	  python ${DRIVER} --nltk)

# compute collocations if not pickled
pickle:
	@mkdir -p pickle

# run LDA w/ scikit or gensim
lda-%: pickle
	@(if hash conda; then source activate text; fi;                    \
	  python ${DRIVER} --lda $(subst lda-,,$@))

optim-%: pickle
	@(if hash conda; then source activate text; fi;                    \
	  python ${DRIVER} --optim $(subst optim-,,$@))

# make cli-<target>
cli-%: pickle
	@(if hash conda; then source activate text; fi;                    \
	  python ${DRIVER} $(subst cli-,,--$@))

# convert org to markdown
convert-%:
	@echo "converting $(subst convert-,,$@)"
	${emacs} $(subst convert-,,$@).org ${EMACS_MD}

# link my code to git repo -- ignore this
links:
	@ln -s $(CURDIR)/Makefile ${GITDIR}/Makefile 2>/dev/null || true
	@for d in ${SRCDIR} ${SETUP} ${ETCDIR}; do                         \
	    mkdir -p ${GITDIR}/$$d;                                        \
	    for i in $$d/*; do                                             \
		echo "Linking $(CURDIR)/$$i -> ${GITDIR}/$$i";             \
	        if [ ! -d $$i ]; then                                      \
	          ln -s $(CURDIR)/$$i ${GITDIR}/$$i 2>/dev/null || true;   \
		fi;                                                        \
	    done;                                                          \
	done

clean:
	$(RM) -r *~ ${SRCDIR}/*.pyc *.pyc

clean-all: clean
	$(RM) -rf ${SRCDIR}/__pycache__ __pycache__
