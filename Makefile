archmm:
	python setup.py install

build:
	python setup.py build_ext --inplace

doc:
	SPHINX_APIDOC_OPTIONS='members,private-members,show-inheritance' sphinx-apidoc -f -M -e -o docs/src/ archmm/ setup.py
	make -C docs/ html

.PHONY: archmm build doc