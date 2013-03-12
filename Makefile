.PHONY: all docs tests build clean web

CC = gcc
LPYTHON = $(shell python-config --includes)
LNUMPY = $(shell python -c "import numpy; print '-I' + numpy.get_include()")

CFLAGS = -lpthread $(LPYTHON) $(LNUMPY)

all: build blir

# stupid hack for now
blir:
	$(CC) $(CFLAGS) -shared -fPIC blaze/blir/datashape.c -o blaze/blir/datashape.o
	$(CC) $(CFLAGS) -shared -fPIC blaze/blir/datashape.o blaze/blir/prelude.c -o blaze/blir/prelude.so

build:
	python setup.py build_ext --inplace

tests:
	nosetests -s -v --detailed blaze

docs:
	cd docs; make html

images:
	cd docs/source/svg; make

web:
	cd web; make html

cleandocs:
	cd docs; make clean

clean:
	python setup.py clean
