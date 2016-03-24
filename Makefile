.PHONY: all

all: proposal.pdf

%.pdf: %.tex
	pdflatex --halt-on-error $^
