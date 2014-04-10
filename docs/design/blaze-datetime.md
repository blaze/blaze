# Date/Time/DateTime

Blaze needs good date/time support for dates and
datetimes to support time series. This document
proposes some design ideas to accomplish that.

There are three main sources of inspiration for this
document, the datetime library in Python's standard
library, the datetime64 type in NumPy, and
the pandas library which uses NumPy's datetime64
storage using nanosecond as the unit together
with the dateutil library for support.

* http://docs.python.org/3.4/library/datetime.html
* http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html
* http://pandas.pydata.org/pandas-docs/stable/timeseries.html
* https://labix.org/python-dateutil / https://code.launchpad.net/dateutil
* http://userguide.icu-project.org/datetime/universaltimescale

The aim of this design is to roughly
model date and datetime as array-programming versions
of Python's standard datetime. While the Python
datetime library has known flaws, they are mostly
regarding completeness. It defines a time zone
object, but does not include standard time zone
implementations, leaving it to third party
libraries like ``pytz`` and ``dateutil`` to fill in
the gap. This does not appear to be a reasonable
approach for Blaze to take.

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

* Always store datetime as a 64-bit ticks
  (100*nanoseconds) offset from midnight of
  January 1, 0001. This is the "universal time
  scale" defined in the ICU library.

http://userguide.icu-project.org/datetime/universaltimescale

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
can't even represent 1492! Microseconds (range of
600000 years) or Ticks (100*nanoseconds, range of
60000 years) seem more reasonable.

```
>>> import pandas as pd
>>> pd.date_range('1492-01-01', '1492-12-31', freq='D')
ValueError: Out of bounds nanosecond timestamp: 1492-01-01 00:00:00
```

## Time Zone

* Will use the Olson tz database, by importing the part which
  knows how to read a compiled database, tzcode, into
  DyND, then having a configuration to point at a build
  version of the database. Where there is an OS version,
  we can point at that, otherwise can point at the version
  inside the ``pytz`` library.

### Resources

* http://nodatime.org/
* http://www.joda.org/joda-time/
* http://www.boost.org/doc/libs/1_55_0/doc/html/date_time/local_time.html
* http://www.ibm.com/developerworks/aix/library/au-aix-posix/index.html
* https://www.gnu.org/software/libc/manual/html_node/TZ-Variable.html
* https://www.iana.org/time-zones
* http://www.cplusplus.com/reference/locale/time_get/get_date/
* https://github.com/ajaxorg/cygwin-builds/blob/master/usr/share/doc/tzcode/Theory
* http://www.timeanddate.com/library/abbreviations/timezones/
* .NET DateTime http://msdn.microsoft.com/en-us/library/bb384267.aspx

* http://pandas.pydata.org/pandas-docs/stable/timeseries.html#time-zone-handling
* http://www.stat.berkeley.edu/classes/s133/dates.html

### Discussion

Time zones are tricky to deal with, for a number of
reasons:

* They get changed over time. When or if daylight
  savings time happens can change, new time zones get
  added, etc. Libraries that support time zones must
  regularly be updated with fresh time zone databases.
* The ISO 8601 string format doesn't specify a
  time zone, it specifies a time zone offset. It's
  possible for such an offset to be the same for
  two different time zones, e.g. one that follows
  daylight savings and one that doesn't.
* There are two main conventions for time zones,
  short string codes like "CST" or "CDT", and
  location based names like "America/Chicago"
  for the central time zone. See the boost
  date_time time zone documentation linked
  above for one library's approach to this.
* The short string codes like "CST" are ambiguous
  in some cases. "CST" might mean "China Standard Time",
  "Central Standard Time", or "Cuba Standard Time".
* CST, for "Central Standard Time" might be for North or
  Central America, with differing daylight savings time
  conventions.
* For POSIX time zone specification, it looks like
  there is usage of GMT and UTC with opposite meanings.

The time zone could be attached to the data, similar
to how Python datetime objects work, or it could be
attached to the type. In Pandas, the time zone
is at the type level, as a property of the
DatetimeIndex object.

### OS-Managed Time Zone Data

The C and C++ libraries don't include time zone
information beyond the current locale, however many
operating systems do have such information accessible.
We would much prefer to use a database managed by
the OS than have to rely on update releases of Blaze
every time any time zone information changes.

On many UNIX systems, including OS X and most Linux
distributions, the Olson tz database is available
from ``/usr/share/zoneinfo``. On the system I looked
at, the files are in the TZif format, which are
generated by the ``zic`` utility included in
the Olson tz database code. As of the present, there
are three possible formats: TZif, TZif2, and TZif3.
The Olson tz database code comes with a reference
implementation for dealing with these zoneinfo files.

In Windows, starting with Windows Vista, there
is information in the registry. This only contains
current time zone information, not historical data,
so does not have the benefits of a full Olson tz
database. From the examples in the blog post linked
below and looking through the registry on a Windows
7 machine, it doesn't look like there's a trivial way
to match the time zone names used here with the POSIX
or Olson time zone names.

http://blogs.msdn.com/b/bclteam/archive/2007/06/07/exploring-windows-time-zones-with-system-timezoneinfo-josh-free.aspx

### Time Zone Attached To Data

* We're going to postpone considering this
  until the case with time zone attached to the dtype
  is functional.

Let's say we want to allow something where we parse
input strings and grab both the datetime value and
the time zone. Some possible formats we might be parsing
are as follows.

```
2001-02-03T04:05         # ISO 8601 with no time zone info
2001-02-03T04:05Z        # ISO 8601 Zulu (UTC) time
2001-02-03T04:05-0600    # ISO 8601 might be CST, GALT, etc.
Sat, 03-Feb-01 04:05 CST # CST
2005-10-21 18:47:22 PDT  # PST during daylight savings
```

If we use ISO 8601 strings as the standard string
representation, we cannot preserve the time zone of
a value when round tripping through a string.

One possibility for attaching time zones to data is to
add another 8 bytes to the value, with a NULL-terminated
string containing either a time zone code listed in
http://www.timeanddate.com/library/abbreviations/timezones/
or a UTC offset (interpreted as a time zone that does not
have daylight savings). A drawback of this is not handling
historical times well, as using the full Olson tz database
would allow. Time zone identifiers in this database are
by contrast relatively long strings, such as
"America/Dawson_Creek".

Systems that behave this way include:

* Python's standard library datetime.datetime.
* In Boost's date_time library, the local_date_time class does this. http://www.boost.org/doc/libs/1_55_0/doc/html/date_time/local_time.html#date_time.local_time.local_date_time
* In the .NET framework, DateTimes are represented as a 62
  bit unsigned integer offset from year 0001 in ticks
  (100 ns increments), with another 2 bits indicating
  whether the data is for UTC, local, or an unspecified
  time zone. This format can represent datetimes from the
  year 0001 to 9999.

### Time Zone Attached To Type

* This is what we will do first.

If we know all the datetimes are in the same time zone,
as is commonly the case in time series, we can attach
the time zone to the type instead of to the data.

Systems that behave this way include:

* Pandas
* Many systems where the time zone is implicit, and must
  be tracked separately by the programmer.

## Code Examples

The following code should work once the system is
completed.

```
>>> from datetime import date, time, datetime, timedelta
>>> import pytz
>>> import blaze, datashape
>>> from blaze import array
>>> from datashape import dshape
```

DataShape creation:

```
>>> dshape('date')
dshape("date")

>>> dshape('time')
dshape("time")
>>> dshape('time[tz="UTC"]')
dshape("time[tz='UTC']")
>>> dshape('time[tz="America/Vancouver"]')
dshape("time[tz='America/Vancouver']")

>>> dshape('datetime')
dshape("datetime")
>>> dshape('datetime[tz="UTC"]')
dshape("datetime[tz='UTC']")
>>> dshape('datetime[tz="America/Vancouver"]')
dshape("datetime[tz='America/Vancouver']")

>>> dshape('units["second"]')
dshape("units['second']")
>>> dshape('units["100*nanosecond", int64]')
dshape('units["100*nanosecond", int64]')
```

Array creation:

```
>>> array(date(2000, 1, 1))
array('2000-01-01',
      dshape='date')
>>> array(datetime(2000, 1, 1, 0, 0), dshape='date')
array('2000-01-01',
      dshape='date')
>>> array(datetime(2000, 1, 1, 5, 0), dshape='date')
ValueError: datetime cannot be converted to a date

>>> array(time(3, 45))
array('03:45',
      dshape='time')
>>> array(time(3, 45, 12, 345))
array('03:45:12.000345',
      dshape='time')
>>> array(time(3, 45, tzinfo=pytz.timezone('America/Vancouver'))
array('03:45',
      dshape='time[tz="America/Vancouver"]')
>>> array('03:45', dshape='time')
array('03:45',
      dshape='time')
>>> array('03:45:30', dshape='time[tz="America/Vancouver"]')
array('03:45:30',
      dshape='time[tz="America/Vancouver"]')

>>> array(datetime(2000, 1, 1))
array('2000-01-01T00:00',
      dshape='datetime')
>>> array(datetime(2000, 1, 1, 3, 45, 30))
array('2000-01-01T03:45:30',
      dshape='datetime')
>>> array(datetime(2000, 1, 1, 3, 45, tzinfo=pytz=timezone('America/Vancouver')))
array('2000-01-01T03:45',
      dshape='datetime[tz="America/Vancouver"]')
>>> array('2000-01-01T03:45', dshape='datetime')
array('2000-01-01T03:45',
      dshape='datetime')
>>> array('2000-01-01T03:45Z', dshape='datetime')
ValueError: Input for 'datetime' cannot specify a time zone
>>> array('2000-01-01T03:45Z', dshape='datetime[tz="America/Vancouver"]')
array('1999-12-31T19:45',
      dshape='datetime[tz="America/Vancouver"]')

>>> array(timedelta(seconds=3))
array(3000000,
      dshape='units["microsecond", int64]')
>>> 3 * blaze.units.second
array(3,
      dshape='units["second"]')
```

DateTime Arithmetic:

```
# NOTE: Python's datetime.time does not support this arithmetic
>>> a = array('2000-01-01', dshape='date')
>>> b = array('03:45', dshape='time')
>>> a + b
array('2000-01-01T03:45',
      dshape='datetime')

>>> a = array('2000-01-05', dshape='date')
>>> b = array('2000-01-01', dshape='date')
>>> a - b
array(4,
      dshape='units["day", int32]')

# NOTE: Python's datetime.time does not support this arithmetic
>>> a = array('03:45', dshape='time')
>>> b = array('02:00', dshape='time')
>>> a - b
array(63000000000,
      dshape='units['100*nanosecond', int64]')

>>> a = array('2000-01-01T03:45', dshape='datetime')
>>> b = array('2000-01-01T02:00', dshape='datetime')
>>> a - b
array(63000000000,
      dshape='units['100*nanosecond', int64]')

>>> a = array('2000-01-01', dshape='date')
>>> a + 100 * blaze.units.day
array('2000-04-10',
      dshape='date')

>>> a = array('2000-01-01T03:45', dshape='datetime')
>>> a + 12345 * blaze.units.millisecond
array('2000-01-01T03:45:12.345',
      dshape='datetime')
```
