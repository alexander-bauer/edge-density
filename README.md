# Motivations and Summary

This repository contains a semester's worth of computer vision research
including much related to automatic "clutter" detection. In this case, clutter
consists of dense clusters of small objects in scenes.

A large and complex component of the Python module presented here is the main
tool, which provides a common entry point for a number of vision routines. To
view its help text and options, use `python -m vision --help` from the root of
this repository.

# Usage

Not all possible uses of the tool are functional or documented.

In general, a detection tool may be invoked as
```
python -m vision [routine] --overlay --draw [input_images...]
```
This will run the detector routine on each image, overlay the result mask on the
original, and draw the image onscreen.

Notably, the routines used for the report in this project are
```
python -m vision corners --overlay --draw data/first_pass/*
python -m vision contour_dense_edges --overlay --draw data/first_pass/*
```

Overlaid results may be colored with `--colorize [red|blue|green|white]`.

They may be saved with `--save [path/to/output/directory]`, and each image will
retain its basename.

To prevent overlaying the results, omit the `--overlay` flag.

If ground truth layers are available, supply `-T` or `--truth`.

# Required Packages

The `vision` module is written in Python 2, and requires OpenCV 2 or 3,
Numpy, and `imutils`. If OpenCV is already installed on your computer with Python 2 bindings, a
virtual environment with the required libraries may be prepared with `make env`.
