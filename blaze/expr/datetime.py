from blaze.expr import Expr, ElemWise
from datashape import dshape, Record, DataShape, Unit
import datashape

class DateTime(ElemWise):
    __slots__ = 'child',

    def __str__(self):
        return '%s.%s' % (str(self.child), type(self).__name__.lower())

    @property
    def schema(self):
        return dshape(self._dtype)

    @property
    def name(self):
        return '%s_%s' % (self.child.name, self.attr)

    @property
    def iscolumn(self):
        return self.child.iscolumn

    @property
    def attr(self):
        return type(self).__name__.lower()


class Date(DateTime):
    _dtype = datashape.date_

def date(expr):
    return Date(expr)

class Year(DateTime):
    _dtype = datashape.int32

def year(expr):
    return Year(expr)

class Month(DateTime):
    _dtype = datashape.int32

def month(expr):
    return Month(expr)

class Day(DateTime):
    _dtype = datashape.int32

def day(expr):
    return Day(expr)

class Time(DateTime):
    _dtype = datashape.time_

def time(expr):
    return Time(Expr)

class Hour(DateTime):
    _dtype = datashape.int32

def hour(expr):
    return Hour(expr)

class Minute(DateTime):
    _dtype = datashape.int32

def minute(expr):
    return Minute(expr)

class Second(DateTime):
    _dtype = datashape.int32

def second(expr):
    return Second(expr)

class Millisecond(DateTime):
    _dtype = datashape.int64

def millisecond(expr):
    return Millisecond(expr)

class Microsecond(DateTime):
    _dtype = datashape.int64

def microsecond(expr):
    return Microsecond(expr)



def isdatelike(ds):
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape):
        ds = ds[0]
    return (isinstance(ds, Unit) or isinstance(ds, Record) and
            len(ds.dict) == 1) and 'date' in str(ds)

from blaze.expr.core import schema_method_list, method_properties

schema_method_list.extend([
    (isdatelike, {year, month, day, hour, minute, date, time,
                  second, millisecond, microsecond})
    ])

method_properties |= {year, month, day, hour, minute, second, millisecond,
        microsecond, date, time}
