REMOTE_ORIGIN=git@github.com:AntoinePassemiers/ArchMM.git

archmm:
	python setup.py install

build:
	python setup.py build_ext --inplace

doc:
	SPHINX_APIDOC_OPTIONS='members,private-members,show-inheritance' sphinx-apidoc -f -M -e -o docs/src/ archmm/ setup.py
	make -C docs/ html

pushdoc:
	cd ../ArchMM-docs/html
	if [ ! -d ".git" ]; then git init; git remote add origin ${REMOTE_ORIGIN}; fi
	git config --global user.email "apassemi@ulb.ac.be"
	git config --global user.name "AntoinePassemiers"
	git add .
	git commit -m "Build the docs"
	git push -f origin HEAD:gh-pages

.PHONY: archmm build doc pushdoc