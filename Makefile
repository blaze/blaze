.PHONY: docs tests build clean

docs:
	cd docs; make html

images:
	cd docs/source/svg; make

cleandocs:
	cd docs; make clean

tests:
	nosetests -s -v --detailed
	#nosetests --rednose -s -v ndtable

build:
	python setup.py build_ext --inplace

clean:
	python setup.py clean
