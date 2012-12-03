.PHONY: all docs tests build clean web

all: build

build:
	python setup.py build_ext --inplace

docs:
	cd docs; make html

images:
	cd docs/source/svg; make

web:
	cd web; make html

cleandocs:
	cd docs; make clean

tests:
	nosetests -s -v --detailed
	#nosetests --rednose -s -v blaze


clean:
	python setup.py clean
