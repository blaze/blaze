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
* https://labix.org/python-dateutil / https://code.launchpad.net/dateutil

The aim of this design is to roughly
model date and datetime as array-programming versions
of Python's standard datetime. While the Python
datetime library has many known flaws, they are mostly
regarding completeness. While it defines a time zone
object, the library does not include standard
time zone implementations, leaving it to third party
libraries like ``pytz`` and ``dateutil`` to fill in
the gap.

## Date

In NumPy, the choice was made to merge date and datetime
into a single object parameterized by unit. Keeping
these separate seems like a better idea, because
``date`` can have much more calendar logic, and doesn't
need to be concerned with time zones, whereas
``datetime`` can operate like the combination of a
``date`` and a ``time``.

### Attributes

```
a.year
a.month
a.day
```

These provide easy access to the components of
the date. They are read-only.

### Methods

``a.replace(year, month, day)``

Returns an array with any or all of year, month,
and day replaced in the dates.

``a.to_struct()``

Converts the dates to structs with type
"{year: int32, month: int16, day: int16}".

``a.strftime(format)``

Formats the dates using the C stdlib strftime function.

``a.weekday()``

Returns the zero-based weekday. Monday is 0, Sunday
is 6.

## DateTime

## No Unit Parameter

* Always store datetime as 64-bit microseconds offset
  from midnight of January 1, 1970.

NumPy parameterizes its datetime64 type with a unit,
ranging from attoseconds up through hours. This
causes the implementation behind the scenes to have
a fair bit of complexity dealing with all the
details as a consequence.

Neither Python's datetime nor the Pandas library
use such a unit parameter. We will go with this
approach, storing all datetimes with a particular
fixed unit.

Pandas chooses nanoseconds as its unit, which means
that it can represent dates from 1678 to 2262. This
seems like a rather limiting choice of default if
the library is to be used for historical dates, it
can't even represent 1492! Microseconds seems more
reasonable, as it gives a range of about 600000 years.
