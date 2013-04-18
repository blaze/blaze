.PHONY: all parser cleanparser clean

all: parser

build: parser
	python setup.py build_ext --inplace

test: build
	python -c "import blir; blir.test()"

parser: cleanparser
	python -m blir.parser

cleanparser:
	-rm -f blir/byacc.py
	-rm -f blir/byacc.pyc
	-rm -f blir/blex.py
	-rm -f blir/blex.pyc

clean: cleanparser
	-rm -Rf build
