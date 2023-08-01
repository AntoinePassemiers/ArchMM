REMOTE_ORIGIN=git@github.com:AntoinePassemiers/ArchMM.git
DOCS_DIR     = docs/

archmm:
	python setup.py install

build:
	python setup.py build_ext --inplace

doc:
	sphinx-apidoc -Mef -o docs/source/apidoc/ archmm/ --separate
	make -C docs/ html

pushdoc:
	cd ${DOCS_DIR}/build/html && \
	if [ ! -d ".git" ]; then git init; git remote add origin ${REMOTE_ORIGIN}; fi && \
	git add . && \
	git commit -m "Build the docs" && \
	git push -f origin HEAD:gh-pages

clean:
	make -C docs/ clean
	python setup.py clean

.PHONY: archmm build doc pushdoc