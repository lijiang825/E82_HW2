SHELL        = /bin/bash
conda        ?= conda
python       ?= python

SETUP        = setup
DATADIR      = data
ETCDIR       = etc
SRCDIR       = src
NLTK_DATADIR = corpora
GITDIR       = $(realpath E82_HW2)

# just for converting from .org to markdown or whatever format
emacs        ?= emacs
EMACS_FLAGS  ?= -batch -Q
EMACS_MD     = ${EMACS_FLAGS} -f org-md-export-to-markdown

all:

run-lda: nltk-data
	@(if hash conda; then source activate text; fi;                    \
	  python ${SRCDIR}/hw2_run.py)

# TODO: where to store data?
# get-data:
# 	@(mkdir -p ${DATADIR} && cd ${DATADIR} &&                          \
# 	wget https://drive.google.com/open?id=180FBOXqxdyvzHihHsg_bAbYS-UF1WrKZ)

# NP: installs packages I've been using with conda creating "text" env
# I don't know if you use conda, but I'll include the environment in case
# if you don't have conda on your path it won't do anything
create-env:
	@if hash conda 2>/dev/null && ! conda env list | grep "envs/text"; \
	then                                                               \
		conda env create -f ${SETUP}/text.yml;                     \
	fi

# download nltk lemma data
nltk-data:
	@(if hash conda; then source activate text; fi;                    \
	  python ${SRCDIR}/hw2_prereqs.py)

# convert org to markdown
convert-%:
	@echo "converting $(subst convert-,,$@)"
	${emacs} $(subst convert-,,$@).org ${EMACS_MD}
# @for i in ${ETCDIR}/*.org; do                                            \
# 	echo "converting $$i -> md";                                       \
# 	${emacs} $$i ${EMACS_MD};                                          \
# done;

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
	$(RM) -rf ${SRCDIR}/__pycache__ __pycache__ ${NLTK_DATADIR}
