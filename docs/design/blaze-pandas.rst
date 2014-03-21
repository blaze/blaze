===============
Blaze to Pandas
===============


Goal
====

Pandas DataFrames have become the de-facto in memory object for many scientific and statistical data analyses
and explorations.  As the PyData eco-system has grown, so too have the demands of the user-base our tools support.
We need blaze to support these growing demands and interoperate with Pandas.


Requirements
============

Export blaze array to Pandas DataFrame


Implementation
==============

Ideally, this should be as simple as::

  x = blaze.array()
  x.as_df()

