.PHONY: all docs tests build clean web

all: build

build:
	# TODO: hack
	#cython ndtable/carray/carrayExtension.pyx
	cython ndtable/engine/driver.pyx
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
	#nosetests --rednose -s -v ndtable


clean:
	python setup.py clean
