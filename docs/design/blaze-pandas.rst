===============
Blaze to Pandas
===============


Goal
====

Pandas DataFrames have become the de-facto in memory object for many scientific and statistical data analyses
 and explorations.  As the PyData eco-system has grown, so too have the demands of the user-base our tools support.
 While  NumPy is the standardize format for representing in memory multi-dimensional arrays, and Pandas is largely
 built on NumPy, Pandas DataFrames offer several niceties and functionality NumPy currently does not offer.  Many
 of the PyData user-base are excited about all the offerings blaze has built and is currently building but downstream
 need/want a Pandas DataFrame.  We therefore propose that Blaze add a mechanism to translate Blaze Arrays to Pandas
 DataFrames

Requirements
============

Export blaze array to Pandas DataFrame


Implementation
==============

As a first past this could be as simple as::

    x = blaze.array()
    y = nd.as_numpy(x, allow_copy=True)
    df = pandas.DataFrame.from_records(y)
