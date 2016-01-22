PyCADU
======

Reader for CADU files as send by the Suomi-NPP satellite.

Usage::

  npp_reader.py [-h] [-p] cadu_file channel [channel ...]

This will generate bw images in the current directory showing the raw data of the requested channels.
At the moment only M-bands work.