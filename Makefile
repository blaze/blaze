.PHONY: all docs tests build clean web

CC = gcc
LPYTHON = $(shell python-config --includes)
LNUMPY = $(shell python -c "import numpy; print '-I' + numpy.get_include()")
CFLAGS = -shared -fPIC $(LPYTHON) $(LNUMPY)
LINK = gcc
PYTHON_LIBS = $(shell python-config --libs)
LFLAGS = -lpthread -shared -fPIC $(PYTHON_LIBS)

UNAME = $(shell uname)

ifeq ($(UNAME), Darwin)
DYLIB_SUFFIX = dylib
endif
ifeq ($(UNAME), Linux)
DYLIB_SUFFIX = so
endif

all: build blir

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

blaze/blir/prelude.$(DYLIB_SUFFIX): blaze/blir/datashape.o blaze/blir/prelude.o
	$(LINK) $(LFLAGS) $< -o $@

# stupid hack for now
blir:	blaze/blir/prelude.$(DYLIB_SUFFIX)

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
