# Build and publish the port-notes book.
#
# Every target runs Python through the `$(PYTHON)` on PATH, never an absolute
# interpreter path. In this directory `.python-version` makes pyenv resolve
# that to the virtualenv it names, so the notebooks execute in that
# environment: their frontmatter asks for `kernelspec: name: python3`, which
# means "whatever python3 kernel the executing Jupyter offers", so the
# environment follows the caller and the caller is fixed here.

PYTHON ?= python
PIP_INSTALL_CMD ?= $(PYTHON) -m pip install
BUILD_DIR = _build/html
JL_DIR = _build/jl

.PHONY: help guard html book github clean rm-ipynb bresenham-fixtures

help:
	@echo "make env       show the interpreter and kernels the notebooks will use"
	@echo "make check     execute every notebook in the TOC, report errors"
	@echo "make check NB=on_lines.md   execute just one"
	@echo "make html      build the book, warnings as errors"
	@echo "make github    build and publish to GitHub Pages"
	@echo "make clean     remove _build and the paired .ipynb files"
	@echo "make bresenham-fixtures   regenerate bresenham_nd_fixtures/*.json"

html:
	# Check for ipynb files in source (should all be paired .md).
	if compgen -G "*.ipynb" 2> /dev/null; then (echo "ipynb files" && exit 1); fi
	$(PYTHON) -c "from jupyter_book.cli.main import main; main()" build -W .

# `book` is an alias for `html`, kept because the notebooks refer to it.
book: html

github: html
	ghp-import -n $(BUILD_DIR) -p -f

clean: rm-ipynb
	rm -rf _build

rm-ipynb:
	rm -rf *.ipynb
