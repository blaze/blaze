.PHONY: all docs tests build clean web

all: build

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
