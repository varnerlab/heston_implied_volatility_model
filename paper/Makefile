TEX = pdflatex
BIB = bibtex
MAIN = main

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib $(wildcard sections/*.tex) $(wildcard sections/figures/*.png)
	-$(TEX) -interaction=nonstopmode $(MAIN)
	-$(BIB) $(MAIN)
	-$(TEX) -interaction=nonstopmode $(MAIN)
	-$(TEX) -interaction=nonstopmode $(MAIN)

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls \
	      $(MAIN).fdb_latexmk $(MAIN).synctex.gz
