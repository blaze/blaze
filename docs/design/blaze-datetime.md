# Date/Time/DateTime

Blaze needs good date/time support for dates and
datetimes to support time series. This document
proposes some design ideas to accomplish that.

There are three sources of inspiration for this
document, the datetime library in Python's standard
library, the datetime64 type in NumPy, and
the pandas library which uses NumPy's datetime64
storage using nanosecond as the unit together
with the dateutil library for support.

* http://docs.python.org/3.4/library/datetime.html
* http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html
* http://pandas.pydata.org/pandas-docs/stable/timeseries.html
* https://code.launchpad.net/dateutil

The aim of this design is to roughly
model date and datetime as array-programming versions
of Python's standard datetime.

## Date

