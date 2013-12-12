# -*- coding: utf-8 -*-
from __future__ import absolute_import

"""
This defines the DataShape type system, with unified
shape and data type.
"""

import sys
import ctypes
import operator
import datetime

import blaze
from ..py2help import _inttypes, _strtypes, unicode

import numpy as np

#------------------------------------------------------------------------
# Type Metaclass
#------------------------------------------------------------------------

# Classes of unit types.
DIMENSION = 1
MEASURE   = 2

class Type(type):
    _registry = {}

    def __new__(meta, name, bases, dct):
        cls = type(name, bases, dct)
        # Don't register abstract classes
        if not dct.get('abstract'):
            Type._registry[name] = cls
        return cls

    @classmethod
    def register(cls, name, type):
        # Don't clobber existing types.
        if name in cls._registry:
            raise TypeError('There is another type registered with name %s'
                            % name)

        cls._registry[name] = type

    @classmethod
    def lookup_type(cls, name):
        return cls._registry[name]

#------------------------------------------------------------------------
# Primitives
#------------------------------------------------------------------------

class Mono(object):
    """
    Monotype are unqualified 0 parameters.

    Each type must be reconstructable using its parameters:

        type(blaze_type)(*type.parameters)
    """
    composite = False
    __metaclass__ = Type

    def __init__(self, *params):
        self.parameters = params

    @property
    def shape(self):
        return ()

    def __len__(self):
        return 1

    def __getitem__(self, key):
        lst = [self]
        return lst[key]

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__,
                           ", ".join(map(repr, self.parameters)))

    # Form for searching signature in meta-method Dispatch Table
    def sigform(self):
        return self

    # Monotypes are their own measure
    @property
    def measure(self):
        return self

    def subarray(self, leading):
        """Returns a data shape object of the subarray with 'leading'
        dimensions removed. In the case of a measure such as CType,
        'leading' must be 0, and self is returned.
        """
        if leading >= 1:
            raise IndexError(('Not enough dimensions in data shape '
                            'to remove %d leading dimensions.') % leading)
        else:
            return self

class Unit(Mono):
    """
    Unit type that does not need to be reconstructed.
    """

#------------------------------------------------------------------------
# Parse Types
#------------------------------------------------------------------------

class Ellipsis(Mono):
    """
    Ellipsis (...). Used to indicate a variable number of dimensions.
    E.g.:

        ..., float32    # float32 array w/ any number of dimensions
        A..., float32   # float32 array w/ any number of dimensions,
                        # associated with type variable A
    """

    def __init__(self, typevar=None):
        self.parameters = (typevar,)

    @property
    def typevar(self):
        return self.parameters[0]

    def __str__(self):
        if self.typevar:
            return str(self.typevar) + '...'
        return '...'

    def __repr__(self):
        return 'Ellipsis("%s")' % (str(self),)

    def __hash__(self):
        return hash('...')

class Null(Unit):
    """
    The null datashape.
    """
    def __str__(self):
        return expr_string('null', None)

class IntegerConstant(Unit):
    """
    An integer which is a parameter to a type constructor. It is itself a
    degenerate type constructor taking 0 parameters.

    ::
        1, int32   # 1 is Fixed
        Range(1,5) # 1 is IntegerConstant

    """
    cls = None

    def __init__(self, i):
        assert isinstance(i, _inttypes)
        self.parameters = (i,)
        self.val = i

    def __str__(self):
        return str(self.val)

    def __eq__(self, other):
        if isinstance(other, _inttypes):
            return self.val == other
        elif isinstance(other, IntegerConstant):
            return self.val == other.val
        else:
            raise TypeError("Cannot compare type %s to type %s" % (type(self), type(other)))

    def __hash__(self):
        return hash(self.val)

class StringConstant(Unit):
    """
    Strings at the level of the constructor.

    ::
        string(3, "utf-8")   # "utf-8" is StringConstant
    """

    def __init__(self, i):
        assert isinstance(i, _strtypes)
        self.parameters = (i,)
        self.val = i

    def __str__(self):
        return repr(self.val)

    def __eq__(self, other):
        if isinstance(other, _strtypes):
            return self.val == other
        elif isinstance(other, StringConstant):
            return self.val == other.val
        else:
            raise TypeError("Cannot compare type %s to type %s" % (type(self), type(other)))

    def __hash__(self):
        return hash(self.val)

class Bytes(Unit):
    """ Bytes type """
    cls = MEASURE

    def __str__(self):
        return 'bytes'

    def __eq__(self, other):
        return isinstance(other, Bytes)

#------------------------------------------------------------------------
# String Type
#------------------------------------------------------------------------

_canonical_string_encodings = {
    u'A' : u'A',
    u'ascii' : u'A',
    u'U8' : u'U8',
    u'utf-8' : u'U8',
    u'utf_8' : u'U8',
    u'utf8' : u'U8',
    u'U16' : u'U16',
    u'utf-16' : u'U16',
    u'utf_16' : u'U16',
    u'utf16' : u'U16',
    u'U32' : u'U32',
    u'utf-32' : u'U32',
    u'utf_32' : u'U32',
    u'utf32' : u'U32'
}

class String(Unit):
    """ String container """
    cls = MEASURE

    def __init__(self, fixlen=None, encoding=None):
        # TODO: Do this constructor better...
        if fixlen is None and encoding is None:
            # String()
            self.fixlen = None
            self.encoding = u'U8'
        elif isinstance(fixlen, _inttypes + (IntegerConstant,)) and \
                        encoding is None:
            # String(fixlen)
            if isinstance(fixlen, IntegerConstant):
                self.fixlen = fixlen.val
            else:
                self.fixlen = fixlen
            self.encoding = u'U8'
        elif isinstance(fixlen, _strtypes + (StringConstant,)) and \
                        encoding is None:
            # String('encoding')
            self.fixlen = None
            if isinstance(fixlen, StringConstant):
                self.encoding = fixlen.val
            else:
                self.encoding = unicode(fixlen)
        elif isinstance(fixlen, _inttypes + (IntegerConstant,)) and \
                        isinstance(encoding, _strtypes + (StringConstant,)):
            # String(fixlen, 'encoding')
            if isinstance(fixlen, IntegerConstant):
                self.fixlen = fixlen.val
            else:
                self.fixlen = fixlen
            if isinstance(encoding, StringConstant):
                self.encoding = encoding.val
            else:
                self.encoding = unicode(encoding)
        else:
            raise ValueError(('Unexpected types to String constructor '
                            '(%s, %s)') % (type(fixlen), type(encoding)))

        # Validate the encoding
        if not self.encoding in _canonical_string_encodings:
            raise ValueError('Unsupported string encoding %s' %
                            repr(self.encoding))

        # Put it in a canonical form
        self.encoding = _canonical_string_encodings[self.encoding]

    def __str__(self):
        if self.fixlen is None and self.encoding == 'U8':
            return 'string'
        elif self.fixlen is not None and self.encoding == 'U8':
            return 'string(%i)' % self.fixlen
        elif self.fixlen is None and self.encoding != 'U8':
            return 'string(%s)' % repr(self.encoding).strip('u')
        else:
            return 'string(%i, %s)' % \
                            (self.fixlen, repr(self.encoding).strip('u'))

    def __repr__(self):
        return ''.join(["ctype(\"", str(self).encode('unicode_escape').decode('ascii'), "\")"])

    def __eq__(self, other):
        if type(other) is String:
            return self.fixlen == other.fixlen and \
                            self.encoding == other.encoding
        else:
            return False

    def __hash__(self):
        return hash((self.fixlen, self.encoding))

#------------------------------------------------------------------------
# Base Types
#------------------------------------------------------------------------

class DataShape(Mono):
    """The Datashape class, implementation for generic composite
    datashape objects"""

    __metaclass__ = Type
    composite = False

    def __init__(self, *parameters, **kwds):
        if len(parameters) > 0:
            self.parameters = tuple(parameters)
            if getattr(self.parameters[-1], 'cls', MEASURE) != MEASURE:
                raise TypeError(('Only a measure can appear on the'
                                ' last position of a datashape, not %s') %
                                repr(self.parameters[-1]))
            for dim in self.parameters[:-1]:
                if getattr(dim, 'cls', DIMENSION) != DIMENSION:
                    raise TypeError(('Only dimensions can appear before the'
                                    ' last position of a datashape, not %s') %
                                    repr(dim))
        else:
            raise ValueError(('the data shape should be constructed from 2 or'
                            ' more parameters, only got %s') % (len(parameters)))
        self.composite = True

        name = kwds.get('name')
        if name:
            self.name = name
            self.__metaclass__._registry[name] = self
        else:
            self.name = None

        ###
        # TODO: Why are low-level concepts like strides and alignment on
        # TODO: the datashape?
        ###

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, index):
        return self.parameters[index]

    def __str__(self):
        if self.name:
            res = self.name
        else:
            res = (', '.join(map(str, self.parameters)))

        return res

    def __eq__(self, other):
        if isinstance(other, DataShape):
            return self.parameters == other.parameters
        elif isinstance(other, Mono):
            return False
        else:
            raise TypeError(('Cannot compare non-datashape '
                            'type %s to datashape') % type(other))

    def __hash__(self):
        return hash(tuple(a for a in self))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return ''.join(["dshape(\"",
                        str(self).encode('unicode_escape').decode('ascii'),
                        "\")"])

    @property
    def shape(self):
        return self.parameters[:-1]

    @property
    def measure(self):
        return self.parameters[-1]

    def sigform(self):
        """Return a data shape object with Fixed dimensions replaced
        by TypeVar dimensions.
        """
        newparams = [TypeVar('i%s'%n) for n in range(len(self.parameters)-1)]
        newparams.append(self.parameters[-1])
        return DataShape(*newparams)

    def subarray(self, leading):
        """Returns a data shape object of the subarray with 'leading'
        dimensions removed.
        """
        if leading >= len(self.parameters):
            raise IndexError(('Not enough dimensions in data shape '
                            'to remove %d leading dimensions.') % leading)
        elif leading in [len(self.parameters) - 1, -1]:
            return self.parameters[-1]
        else:
            return DataShape(*self.parameters[leading:])

#------------------------------------------------------------------------
# Categorical
#------------------------------------------------------------------------

class Enum(DataShape):

    def __init__(self, name, *elts):
        self.parameters = (name,) + elts
        self.name = name
        self.elts = elts

    def __str__(self):
        if self.name:
            return 'Enum(%s, %s)' % (self.name, ','.join(map(str, self.elts)))
        else:
            return 'Enum(%s)' % ','.join(map(str, self.elts))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

#------------------------------------------------------------------------
# Missing
#------------------------------------------------------------------------

class Option(DataShape):
    """
    Measure types which may or may not hold data. Makes no
    indication of how this is implemented in memory.
    """

    def __init__(self, *params):
        if len(params) != 1:
            raise TypeError('Option only takes 1 argument')

        if not params[0].cls == MEASURE:
            raise TypeError('Option only takes measure argument')

        self.parameters = params
        self.ty = params[0]

    def __str__(self):
        return 'Option(%s)' % str(self.ty)

    def __repr__(self):
        return str(self)

#------------------------------------------------------------------------
# CType
#------------------------------------------------------------------------

class CType(Unit):
    """
    Symbol for a sized type mapping uniquely to a native type.
    """
    cls = MEASURE

    def __init__(self, name, itemsize, alignment):
        self.name = name
        self._itemsize = itemsize
        self._alignment = alignment
        Type.register(name, self)
        self.parameters = (name,)

    @classmethod
    def from_numpy_dtype(self, dt):
        """
        From Numpy dtype.

        >>> from blaze.datashape import CType
        >>> from numpy import dtype
        >>> CType.from_numpy_dtype(dtype('int32'))
        ctype("int32")
        """
        return Type._registry[dt.name]

    @property
    def itemsize(self):
        """The size of one element of this type."""
        return self._itemsize

    @property
    def c_itemsize(self):
        """The size of one element of this type, with C-contiguous storage."""
        return self._itemsize

    @property
    def c_alignment(self):
        """The alignment of one element of this type."""
        return self._alignment

    def to_numpy_dtype(self):
        """
        To Numpy dtype.
        """
        # Fixup the complex type to how numpy does it
        s = self.name
        s = {'cfloat32':'complex64', 'cfloat64':'complex128'}.get(s, s)
        return np.dtype(s)

    def __str__(self):
        return self.name

    def __repr__(self):
        return ''.join(["ctype(\"", str(self).encode('unicode_escape').decode('ascii'), "\")"])

    def __eq__(self, other):
        if type(other) is CType:
            return self.name == other.name
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

#------------------------------------------------------------------------
# Dimensions
#------------------------------------------------------------------------

class Fixed(Unit):
    """
    Fixed dimension.
    """
    cls = DIMENSION

    def __init__(self, i):
        assert isinstance(i, _inttypes)

        if i < 0:
            raise ValueError('Fixed dimensions must be positive')

        self.val = i
        self.parameters = (self.val,)

    def __index__(self):
        return self.val

    def __int__(self):
        return self.val

    def __eq__(self, other):
        if type(other) is Fixed:
            return self.val == other.val
        elif isinstance(other, _inttypes):
            return self.val == other
        else:
            return False

    def __hash__(self):
        return hash(self.val)

    def __str__(self):
        return str(self.val)

class Var(Unit):
    """ Variable dimension """
    cls = DIMENSION

    def __str__(self):
        return 'var'

    def __eq__(self, other):
        return isinstance(other, Var)

    def __hash__(self):
        return id(Var)

#------------------------------------------------------------------------
# Variable
#------------------------------------------------------------------------

class TypeVar(Unit):
    """
    A free variable in the signature. Not user facing.
    """
    # cls could be MEASURE or DIMENSION, depending on context

    def __init__(self, symbol):
        if symbol.startswith("'"):
            symbol = symbol[1:]
        self.symbol = symbol
        self.parameters = (symbol,)

    def __repr__(self):
        return "TypeVar(%s)" % (str(self),)

    def __str__(self):
        return str(self.symbol)

    # All TypeVariables compare equal
    # dshape('M,int32') = dshape('N,int32')
    # def __eq__(self, other):
    #     if not isinstance(other, TypeVar):
    #         return False
    #     else:
    #         return True

    # def __hash__(self):
    #     return hash(self.__class__)


class Implements(Mono):
    """
    Type representing a constraint on the subtype term (which must be a
    TypeVar), namely that it must belong to a given type set.
    """

    @property
    def typevar(self):
        return self.parameters[0]

    @property
    def typeset(self):
        return self.parameters[1]

    def __repr__(self):
        return '%s : %s' % (self.typevar, self.typeset.name)


class Range(Mono):
    """
    Range type representing a bound or unbound interval of
    of possible Fixed dimensions.
    """
    cls = DIMENSION

    def __init__(self, a, b=False):
        if isinstance(a, _inttypes):
            self.a = a
        elif isinstance(a, IntegerConstant):
            self.a = a.val
        else:
            raise TypeError('Expected integer for parameter a, not %s' % type(a))

        if isinstance(b, _inttypes):
            self.b = b
        elif b is False or b is None:
            self.b = b
        elif isinstance(b, IntegerConstant):
            self.b = b.val
        else:
            raise TypeError('Expected integer for parameter b, not %s' % type(b))

        if a and b:
            assert self.a < self.b, 'Must have upper < lower'
        self.parameters = (self.a, self.b)

    @property
    def upper(self):
        # Just upper bound
        if self.b == False:
            return self.a

        # No upper bound case
        elif self.b == None:
            return float('inf')

        # Lower and upper bound
        else:
            return self.b

    @property
    def lower(self):
        # Just upper bound
        if self.b == False:
            return 0

        # No upper bound case
        elif self.b == None:
            return self.a

        # Lower and upper bound
        else:
            return self.a

    def __eq__(self, other):
        if not isinstance(other, Range):
            raise TypeError("Cannot compare type %s to type %s" % (type(self), type(other)))

        else:
            return self.a == other.a and self.b == other.b

    def __hash__(self):
        return hash((self.a, self.b))

    def __str__(self):
        return expr_string('Range', [self.lower, self.upper])

#------------------------------------------------------------------------
# Function signatures
#------------------------------------------------------------------------

class Function(Mono):
    """
    Used for function signatures.
    """
    def __init__(self, *parameters):
        self.parameters = parameters

    @property
    def restype(self):
        return self.parameters[-1]

    @property
    def argtypes(self):
        return self.parameters[:-1]

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                self.parameters == other.parameters)

    def __ne__(self, other):
        return not self == other

    # def __repr__(self):
    #     return " -> ".join(map(repr, self.parameters))

    def __str__(self):
        return " -> ".join(map(str, self.parameters))

#------------------------------------------------------------------------
# Record Types
#------------------------------------------------------------------------

class Record(Mono):
    """
    A composite data structure of ordered fields mapped to types.
    """
    cls = MEASURE

    def __init__(self, fields):
        """
        Parameters
        ----------
        fields : list/OrderedDict of (name, type) entries
            The fields which make up the record.
        """
        # This is passed in with a OrderedDict so field order is
        # preserved. Using RecordDecl there is some magic to also
        # ensure that the fields align in the order they are
        # declared.
        self.__fdict = dict(fields)
        self.__fnames = [f[0] for f in fields]
        self.__ftypes = [f[1] for f in fields]
        self.parameters = (fields,)

    @property
    def fields(self):
        return self.__fdict

    @property
    def names(self):
        return self.__fnames

    @property
    def types(self):
        return self.__ftypes

    def to_numpy_dtype(self):
        """
        To Numpy record dtype.
        """
        dk = self.__fnames
        dv = map(to_numpy_dtype, self.__ftypes)
        return np.dtype(zip(dk, dv))

    def __getitem__(self, key):
        return self.__fdict[key]

    def __eq__(self, other):
        if isinstance(other, Record):
            return self.__fdict == other.__fdict
        else:
            return False

    def __hash__(self):
        return hash(self.__fdict)

    def __str__(self):
        return record_string(self.__fnames, self.__ftypes)

    def __repr__(self):
        return ''.join(["dshape(\"", str(self).encode('unicode_escape').decode('ascii'), "\")"])

#------------------------------------------------------------------------
# JSON
#------------------------------------------------------------------------

class JSON(Mono):
    """ JSON measure """
    cls = MEASURE

    def __init__(self):
        self.parameters = ()

    def __str__(self):
        return 'json'

    def __eq__(self, other):
        return isinstance(other, JSON)

#------------------------------------------------------------------------
# Generic type constructors
#------------------------------------------------------------------------

class TypeConstructor(type):
    """
    Generic type constructor.

    Attributes:
    ===========
        n: int
            number of parameters

        flags: [{str: object}]
            flag for each parameter. Built-in flags include:

                * 'coercible': True/False. The default is False
    """

    def __new__(cls, name, n, flags, is_vararg=False):
        def __init__(self, *params):
            if len(params) != n:
                if not (is_vararg and len(params) >= n):
                    raise TypeError(
                        "Expected %d parameters for constructor %s, got %d" % (
                            n, name, len(params)))
            self.parameters = params

        def __eq__(self, other):
            return (isinstance(other, type(self)) and
                    self.parameters == other.parameters and
                    self.flags == other.flags)

        def __hash__(self):
            return hash((name, n, self.parameters))

        def __str__(self):
            return "%s[%s]" % (name, ", ".join(map(str, self.parameters)))

        d = {
            '__init__': __init__,
            '__repr__': __str__,
            '__str__': __str__,
            '__eq__': __eq__,
            '__ne__': lambda self, other: not (self == other),
            '__hash__': __hash__,
            'flags': flags,
        }
        self = super(TypeConstructor, cls).__new__(cls, name, (Mono,), d)

        self.name = name
        self.n = n
        self.flags = flags
        return self

    def __init__(self, *args, **kwds):
        pass # Swallow arguments

    def __eq__(cls, other):
        return (isinstance(other, TypeConstructor) and
                cls.name == other.name and cls.n == other.n and
                cls.flags == other.flags)

    def __ne__(cls, other):
        return not (cls == other)

    def __hash__(cls):
        return hash((cls.name, cls.n))

#------------------------------------------------------------------------
# Unit Types
#------------------------------------------------------------------------

bool_      = CType('bool', 1, 1)
char       = CType('char', 1, 1)

int8       = CType('int8', 1, 1)
int16      = CType('int16', 2, ctypes.alignment(ctypes.c_int16))
int32      = CType('int32', 4, ctypes.alignment(ctypes.c_int32))
int64      = CType('int64', 8, ctypes.alignment(ctypes.c_int64))

uint8      = CType('uint8', 1, 1)
uint16     = CType('uint16', 2, ctypes.alignment(ctypes.c_uint16))
uint32     = CType('uint32', 4, ctypes.alignment(ctypes.c_uint32))
uint64     = CType('uint64', 8, ctypes.alignment(ctypes.c_uint64))

float16    = CType('float16', 2, ctypes.alignment(ctypes.c_uint16))
float32    = CType('float32', 4, ctypes.alignment(ctypes.c_float))
float64    = CType('float64', 8, ctypes.alignment(ctypes.c_double))
#float128   = CType('float128', 16)

cfloat32  = CType('cfloat32', 8, ctypes.alignment(ctypes.c_float))
cfloat64 = CType('cfloat64', 16, ctypes.alignment(ctypes.c_double))
Type.register('complex64', cfloat32)
complex64  = cfloat32
Type.register('complex128', cfloat64)
complex128 = cfloat64
#complex256 = CType('complex256', 32)

timedelta64 = CType('timedelta64', 8, ctypes.alignment(ctypes.c_int64))
datetime64 = CType('datetime64', 8, ctypes.alignment(ctypes.c_int64))

c_byte = int8
c_short = int16
c_int = int32
c_longlong = int64

c_ubyte = uint8
c_ushort = uint16
c_ulonglong = uint64

if ctypes.sizeof(ctypes.c_long) == 4:
    c_long = int32
    c_ulong = uint32
else:
    c_long = int64
    c_ulong = uint64

if ctypes.sizeof(ctypes.c_void_p) == 4:
    intptr = c_ssize_t = int32
    uintptr = c_size_t = uint32
else:
    intptr = c_ssize_t = int64
    uintptr = c_size_t = uint64
Type.register('intptr', intptr)
Type.register('uintptr', uintptr)

c_half = float16
c_float = float32
c_double = float64
# TODO: Deal with the longdouble == one of float64/float80/float96/float128 situation
#c_longdouble = float128

half = float16
single = float32
double = float64

# TODO: the semantics of these are still being discussed
int_ = int32
float_ = float32

void = CType('void', 0, 1)
object_ = pyobj = CType('object',
                ctypes.sizeof(ctypes.py_object),
                ctypes.alignment(ctypes.py_object))

na = Null
NullRecord = Record(())
bytes_ = Bytes()

string = String()
json = JSON()

Type.register('float', c_float)
Type.register('double', c_double)

Type.register('bytes', bytes_)

Type.register('string', String())

#------------------------------------------------------------------------
# NumPy Compatibility
#------------------------------------------------------------------------

class NotNumpyCompatible(Exception):
    """
    Raised when we try to convert a datashape into a NumPy dtype
    but it cannot be ceorced.
    """
    pass

def to_numpy_dtype(ds):
    """ Throw away the shape information and just return the
    measure as NumPy dtype instance."""
    return to_numpy(ds[-1])

def to_numpy(ds):
    """
    Downcast a datashape object into a Numpy (shape, dtype) tuple if
    possible.

    >>> from blaze.datashape import dshape, to_numpy
    >>> to_numpy(dshape('5, 5, int32'))
    ((5, 5), dtype('int32'))
    """

    if isinstance(ds, CType):
        return ds.to_numpy_dtype()

    shape = tuple()
    dtype = None

    #assert isinstance(ds, DataShape)

    # The datashape dimensions
    for dim in ds[:-1]:
        if isinstance(dim, IntegerConstant):
            shape += (dim,)
        elif isinstance(dim, Fixed):
            shape += (dim.val,)
        elif isinstance(dim, TypeVar):
            shape += (-1,)
        else:
            raise NotNumpyCompatible('Datashape dimension %s is not NumPy-compatible' % dim)

    # The datashape measure
    msr = ds[-1]
    if isinstance(msr, CType):
        dtype = msr.to_numpy_dtype()
    elif isinstance(msr, Record):
        dtype = msr.to_numpy_dtype()
    else:
        raise NotNumpyCompatible('Datashape measure %s is not NumPy-compatible' % msr)

    if type(dtype) != np.dtype:
        raise NotNumpyCompatible('Internal Error: Failed to produce NumPy dtype')
    return (shape, dtype)


def from_numpy(shape, dt):
    """
    Upcast a (shape, dtype) tuple if possible.

    >>> from blaze.datashape import from_numpy
    >>> from numpy import dtype
    >>> from_numpy((5,5), dtype('int32'))
    dshape("5, 5, int32")
    """
    dtype = np.dtype(dt)

    if dtype.kind == 'S':
        measure = String(dtype.itemsize, 'A')
    elif dtype.kind == 'U':
        measure = String(dtype.itemsize / 4, 'U8')
    elif dtype.fields:
        rec = [(a,CType.from_numpy_dtype(b[0])) for a,b in dtype.fields.items()]
        measure = Record(rec)
    else:
        measure = CType.from_numpy_dtype(dtype)

    if shape == ():
        return measure
    else:
        return DataShape(*tuple(map(Fixed, shape))+(measure,))

#------------------------------------------------------------------------
# Python Compatibility
#------------------------------------------------------------------------

def typeof(obj):
    """
    Return a datashape ctype for a python scalar.
    """
    if isinstance(obj, blaze.Array):
        return obj.dshape
    elif isinstance(obj, np.ndarray):
        return from_numpy(obj.shape, obj.dtype)
    elif isinstance(obj, _inttypes):
        return DataShape(int_)
    elif isinstance(obj, float):
        return DataShape(double)
    elif isinstance(obj, complex):
        return DataShape(complex128)
    elif isinstance(obj, _strtypes):
        return DataShape(string)
    elif isinstance(obj, datetime.timedelta):
        return DataShape(timedelta64)
    elif isinstance(obj, datetime.datetime):
        return DataShape(datetime64)
    else:
        return DataShape(pyobj)

#------------------------------------------------------------------------
# Printing
#------------------------------------------------------------------------

def expr_string(spine, const_args, outer=None):
    if not outer:
        outer = '()'

    if const_args:
        return str(spine) + outer[0] + ','.join(map(str,const_args)) + outer[1]
    else:
        return str(spine)

def record_string(fields, values):
    # Prints out something like this:
    #   {a : int32, b: float32, ... }
    body = ''
    count = len(fields)

    for i, (k,v) in enumerate(zip(fields,values)):
        if (i+1) == count:
            body += '%s : %s' % (k,v)
        else:
            body += '%s : %s; ' % (k,v)
    return '{ ' + body + ' }'

#------------------------------------------------------------------------
# Type variables
#------------------------------------------------------------------------

def free(ds):
    """
    Return the free variables (TypeVar) of a blaze type (Mono).
    """
    if isinstance(ds, TypeVar):
        return [ds]
    elif isinstance(ds, Mono) and not isinstance(ds, Unit):
        result = []
        for x in ds.parameters:
            result.extend(free(x))
        return result
    else:
        return []

def type_constructor(ds):
    """
    Get the type constructor for the blaze type (Mono).
    The type constructor indicates how types unify (see unification.py).
    """
    return type(ds)
