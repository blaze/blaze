SVGFILES := $(wildcard *.svg)
PNGFILES := $(patsubst %.svg, %.png, $(SVGFILES))

all: $(PNGFILES)

%.png: %.svg
	inkscape -f $< -e $@

.PHONY: all
