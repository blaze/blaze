.PHONY: all parser cleanparser clean

all: parser

build:
	python setup.py build_ext --inplace

test: build
	python -m unittest discover blaze/blz/tests
	python -m unittest discover blaze/aterm/tests

parser: cleanparser
	python -m blaze.blir.parser

cleanparser:
	-rm -f blaze/blir/byacc.py
	-rm -f blaze/blir/byacc.pyc
	-rm -f blaze/blir/blex.py
	-rm -f blaze/blir/blex.pyc

clean: cleanparser
	-rm -Rf build
