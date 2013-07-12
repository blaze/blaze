.PHONY: all clean

all: build

build:
	python setup.py build_ext --inplace

test: build
	python -m unittest discover blaze/blz/tests

clean:
	-rm -Rf build
