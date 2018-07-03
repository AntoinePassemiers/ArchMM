archmm:
	python setup.py install

build:
	python setup.py build_ext --inplace

.PHONY: archmm build