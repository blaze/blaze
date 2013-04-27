# -*- coding: utf-8 -*-

"""
This defines the DataShape type system. The unification of shape and
dtype.
"""

import operator
import numpy as np
import datetime

instanceof = lambda T: lambda X: isinstance(X, T)

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

    @staticmethod
    def register(name, type):
        # Don't clobber existing types.
        if name in Type._registry:
            raise TypeError('There is another type registered with name %s'\
                % name)

        Type._registry[name] = type

    @classmethod
    def lookup_type(cls, name):
        return cls._registry[name]

#------------------------------------------------------------------------
# Primitives
#------------------------------------------------------------------------

class Mono(object):
    """
    Monotype
    """
    composite = False
    __metaclass__ = Type

    def __init__(self, *params):
        self.parameters = params

    def __mul__(self, other):
        if not isinstance(other, (DataShape, Mono)):
            if type(other) is int:
                other = IntegerConstant(other)
            else:
                raise NotImplementedError()
        return product(self, other)

    def __rmul__(self, other):
        if not isinstance(other, (DataShape, Mono)):
            if type(other) is int:
                other = IntegerConstant(other)
            else:
                raise NotImplementedError()
        return product(other, self)


class Poly(object):
    """
    Polytype
    """
    def __init__(self, qualifier, *params):
        self.parameters = params

#------------------------------------------------------------------------
# Parse Types
#------------------------------------------------------------------------

class Null(Mono):
    """
    The null datashape.
    """
    def __str__(self):
        return expr_string('null', None)

class IntegerConstant(Mono):
    """
    Integers at the level of constructor it just means integer in the
    sense of of just an integer value to a constructor.

    ::

        1, int32   # 1 is Fixed
        Range(1,5) # 1 is IntegerConstant

    """

    def __init__(self, i):
        assert isinstance(i, int)
        self.val = i

    def __str__(self):
        return str(self.val)

class StringConstant(Mono):
    """
    Strings at the level of the constructor.

    ::

        string(3, "utf-8")   # "utf-8" is StringConstant
    """

    def __init__(self, i):
        assert isinstance(i, (str, unicode))
        self.val = i

    def __str__(self):
        return repr(self.val)

class Dynamic(Mono):
    """
    The dynamic type allows an explicit upcast and downcast from any
    type to ``?``.
    """

    def __str__(self):
        return '?'

    def __repr__(self):
        # need double quotes to form valid aterm, also valid Python
        return ''.join(["dshape(\"", str(self).encode('unicode_escape'), "\")"])

class Top(Mono):
    """ The top type """

    def __str__(self):
        return 'top'

    def __repr__(self):
        # emulate numpy
        return ''.join(["dshape(\"", str(self), "\")"])

class Blob(Mono):
    """ Blob type, large variable length string """
    cls = MEASURE

    def __str__(self):
        return 'blob'

    def __repr__(self):
        # need double quotes to form valid aterm, also valid Python
        return ''.join(["dshape(\"", str(self).encode('unicode_escape'), "\")"])

class Varchar(Mono):
    """ Blob type, small variable length string """
    cls = MEASURE


    def __init__(self, maxlen):
        assert isinstance(maxlen, IntegerConstant)
        self.maxlen = maxlen.val

    def __str__(self):
        return 'varchar(maxlen=%i)' % self.maxlen

    def __repr__(self):
        return expr_string('varchar', [self.maxlen])

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
class String(Mono):
    """ String container """
    cls = MEASURE

    def __init__(self, fixlen=None, encoding=None):
        if fixlen is None and encoding is None:
            # String()
            self.fixlen = None
            self.encoding = u'U8'
        elif isinstance(fixlen, (int, long, IntegerConstant)) and encoding is None:
            # String(fixlen)
            if isinstance(fixlen, IntegerConstant):
                self.fixlen = fixlen.val
            else:
                self.fixlen = fixlen
            self.encoding = u'U8'
        elif isinstance(fixlen, (str, unicode, StringConstant)) and encoding is None:
            # String('encoding')
            self.fixlen = None
            if isinstance(fixlen, StringConstant):
                self.encoding = fixlen.val
            else:
                self.encoding = unicode(fixlen)
        elif isinstance(fixlen, (int, long, IntegerConstant)) and \
                        isinstance(encoding, (str, unicode, StringConstant)):
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
            raise ValueError('Unexpected types to String constructor (%s, %s)' %
                            (type(fixlen), type(encoding)))
        # Validate the encoding
        if not self.encoding in _canonical_string_encodings:
            raise ValueError('Unsupported string encoding %s' % repr(self.encoding))
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
            return 'string(%i, %s)' % (self.fixlen, repr(self.encoding).strip('u'))

    def __repr__(self):
        # need double quotes to form valid aterm, also valid
        # Python
        return ''.join(["dshape(\"", str(self).encode('unicode_escape'), "\")"])

    def __eq__(self, other):
        if type(other) is String:
            return self.fixlen == other.fixlen and self.encoding == other.encoding
        else:
            return False

#------------------------------------------------------------------------
# Base Types
#------------------------------------------------------------------------

# TODO: figure out consistent spelling for this
#
#   - DataShape
#   - Datashape

class DataShape(Mono):
    """ The Datashape class, implementation for generic composite
    datashape objects """

    __metaclass__ = Type
    composite = False

    def __init__(self, parameters=None, name=None):

        if type(parameters) is DataShape:
            self.parameters = parameters
        elif len(parameters) > 0:
            self.parameters = tuple(flatten(parameters))
            if getattr(self.parameters[-1], 'cls', MEASURE) != MEASURE:
                raise TypeError('Only a measure can appear on the last position of a datashape, not %s' % repr(self.parameters[-1]))
            for dim in self.parameters[:-1]:
                if getattr(dim, 'cls', DIMENSION) != DIMENSION:
                    raise TypeError('Only dimensions can appear before the last position of a datashape, not %s' % repr(dim))
            self.composite = True
        else:
            self.parameters = tuple()
            self.composite = False

        if name:
            self.name = name
            self.__metaclass__._registry[name] = self
        else:
            self.name = None

    def __getitem__(self, index):
        return self.parameters[index]

    def __str__(self):
        if self.name:
            return self.name
        else:
            return (', '.join(map(str, self.parameters)))

    def _equal(self, other):
        """ Structural equality """
        return all(a==b for a,b in zip(self, other))

    def __eq__(self, other):
        if type(other) is DataShape:
            return self._equal(other)
        else:
            raise TypeError('Cannot compare non-datashape type %s to datashape' % type(other))

    def __repr__(self):
        # need double quotes to form valid aterm, also valid Python
        return ''.join(["dshape(\"", str(self).encode('unicode_escape'), "\")"])

    @property
    def shape(self):
        return self.parameters[:-1]

    # Alternative constructors
    # ------------------------

    def __or__(self, other):
        return Either(self, other)

    def __mul__(self, other):
        if not isinstance(other, (DataShape, Mono)):
            if type(other) is int:
                other = IntegerConstant(other)
            else:
                raise NotImplementedError()
        return product(self, other)

    def __rmul__(self, other):
        if not isinstance(other, (DataShape, Mono)):
            if type(other) is int:
                other = IntegerConstant(other)
            else:
                raise NotImplementedError()
        return product(other, self)

class Atom(DataShape):
    """
    Atoms for arguments to constructors of types, not types in
    and of themselves.
    """
    abstract = True

    def __init__(self, *parameters):
        self.parameters = parameters

    def __str__(self):
        clsname = self.__class__.__name__
        return expr_string(clsname, self.parameters)

    def __repr__(self):
        return str(self)

#------------------------------------------------------------------------
# CType
#------------------------------------------------------------------------

NATIVE = '='
LITTLE = '<'
BIG    = '>'

class CType(DataShape):
    """
    Symbol for a sized type mapping uniquely to a native type.
    """
    cls = MEASURE

    def __init__(self, ctype, size=None, byteorder=None):
        if size:
            assert 1 <= size < (2**23-1)
            label = ctype + str(size)
            self.parameters = [label]
            self.name = label
            self.byteorder = byteorder or LITTLE
            Type.register(label, self)
        else:
            self.parameters = [ctype]
            self.name = ctype
            Type.register(ctype, self)

    @classmethod
    def from_str(self, s):
        """
        To Numpy dtype.

        >>> CType.from_str('int32')
        int32
        """
        return Type._registry[s]

    @classmethod
    def from_numpy_dtype(self, dt):
        """
        From Numpy dtype.

        >>> CType.from_numpy_dtype(dtype('int32'))
        int32
        """
        return Type._registry[dt.name]

    def size(self):
        # TODO: no cheating!
        return np.dtype(self.name).itemsize

    def to_struct(self):
        """
        To struct code.
        """
        return np.dtype(self.name).char

    def to_dtype(self):
        """
        To Numpy dtype.
        """
        # special cases because of NumPy weirdness
        # >>> dtype('i')
        # dtype('int32')
        # >>> dtype('int')
        # dtype('int64')
        if self.name == "int":
            return np.dtype("i")
        if self.name == "float":
            return np.dtype("f")
        return np.dtype(self.name)

    def __str__(self):
        return str(self.parameters[0])

    def __eq__(self, other):
        if type(other) is CType:
            return self.parameters[0] == other.parameters[0]
        else:
            return False

    @property
    def type(self):
        raise NotImplementedError()

    @property
    def kind(self):
        raise NotImplementedError()

    @property
    def char(self):
        raise NotImplementedError()

    @property
    def num(self):
        raise NotImplementedError()

    @property
    def str(self):
        raise NotImplementedError()

#------------------------------------------------------------------------
# Fixed
#------------------------------------------------------------------------

class Fixed(Atom):
    """
    Fixed dimension.
    """
    cls = DIMENSION

    def __init__(self, i):
        assert isinstance(i, (int, long))

        if i < 0:
            raise ValueError, 'Fixed dimensions must be positive'

        self.val = i
        self.parameters = [self.val]

    def __index__(self):
        return self.val

    def __int__(self):
        return self.val

    def __eq__(self, other):
        if type(other) is Fixed:
            return self.val == other.val
        else:
            return False

    def __gt__(self, other):
        if type(other) is Fixed:
            return self.val > other.val
        else:
            return False

    def __str__(self):
        return str(self.val)

#------------------------------------------------------------------------
# Variable
#------------------------------------------------------------------------

class TypeVar(Atom):
    """
    A free variable in the signature. Not user facing.
    """
    # cls could be MEASURE or DIMENSION, depending on context

    def __init__(self, symbol):
        if symbol.startswith("'"):
            symbol = symbol[1:]
        self.symbol = symbol
        self.parameters = [symbol]

    def __str__(self):
        # Use the F# notation
        return str(self.symbol)
        # return "'" + str(self.symbol)

class Range(Atom):
    """
    Range type representing a bound or unbound interval of
    of possible Fixed dimensions.
    """
    cls = DIMENSION

    def __init__(self, a, b=False):
        if isinstance(a, (int, long)):
            self.a = a
        elif isinstance(a, IntegerConstant):
            self.a = a.val
        else:
            raise TypeError('Expected integer for parameter a, not %s' % type(a))

        if isinstance(b, (int, long)):
            self.b = b
        elif b is False or b is None:
            self.b = b
        elif isinstance(b, IntegerConstant):
            self.b = b.val
        else:
            raise TypeError('Expected integer for parameter b, not %s' % type(b))

        if a and b:
            assert self.a < self.b, 'Must have upper < lower'
        self.parameters = [self.a, self.b]

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

    def __str__(self):
        return expr_string('Range', [self.lower, self.upper])

#------------------------------------------------------------------------
# Aggregate
#------------------------------------------------------------------------

class Either(Atom):
    """
    A datashape for tagged union of values that can take on two
    different, but fixed, types called tags ``left`` and ``right``. The
    tag deconstructors for this type are :func:`inl` and :func:`inr`.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.parameters = [a,b]

class Option(Atom):
    """
    A sum type for nullable measures unit types. Can be written
    as a tagged union with with ``left`` as ``null`` and
    ``right`` as a measure.
    """
    cls = MEASURE

    def __init__(self, ty):
        self.parameters = [ty]

class Factor(Atom):
    """
    A finite enumeration of Fixed dimensions.
    """

    def __str__(self):
        # Use c-style enumeration syntax
        return expr_string('', self.parameters, '{}')

class Union(Atom):
    """
    A untagged union is a datashape for a value that may hold
    several but fixed datashapes.
    """

    def __str__(self):
        return expr_string('', self.parameters, '{}')

class Record(DataShape):
    """
    A composite data structure of ordered fields mapped to types.
    """
    cls = MEASURE

    def __init__(self, fields):
        # This is passed in with a OrderedDict so field order is
        # preserved. Using RecordDecl there is some magic to also
        # ensure that the fields align in the order they are
        # declared.
        self.__d = dict(fields)
        self.__k = [f[0] for f in fields]
        self.__v = [f[1] for f in fields]
        self.parameters = [fields]

    @property
    def fields(self):
        return self.__d

    @property
    def names(self):
        return self.__k

    def to_dtype(self):
        """
        To Numpy record dtype.
        """
        dk = self.__k
        dv = map(to_dtype, self.__v)
        return np.dtype(zip(dk, dv))

    def __getitem__(self, key):
        return self.__d[key]

    def __eq__(self, other):
        if isinstance(other, Record):
            return self.__d == other.__d
        else:
            return False

    def __iter__(self):
        return zip(self.__k, self.__v)

    def __len__(self):
        return len(self.__k)

    def __str__(self):
        return record_string(self.__k, self.__v)

    def __repr__(self):
        # need double quotes to form valid aterm, also valid Python
        return ''.join(["dshape(\"", str(self).encode('unicode_escape'), "\")"])

#------------------------------------------------------------------------
# Constructions
#------------------------------------------------------------------------

def product(A, B):
    if A.composite and B.composite:
        f = A.parameters
        g = B.parameters

    elif A.composite:
        f = A.parameters
        g = (B,)

    elif B.composite:
        f = (A,)
        g = B.parameters

    else:
        f = (A,)
        g = (B,)

    return DataShape(parameters=(f+g))

def inr(ty):
    return ty.a

def inl(ty):
    return ty.b

#------------------------------------------------------------------------
# Unit Types
#------------------------------------------------------------------------

bool_      = CType('bool')
char       = CType('char')

int8       = CType('int', 8)
int16      = CType('int', 16)
int32      = CType('int', 32)
int64      = CType('int', 64)

uint8      = CType('uint',  8)
uint16     = CType('uint', 16)
uint32     = CType('uint', 32)
uint64     = CType('uint', 64)

float16    = CType('float', 16)
float32    = CType('float', 32)
float64    = CType('float', 64)
float128   = CType('float', 128)

complex64  = CType('complex' , 64)
complex128 = CType('complex', 128)
complex256 = CType('complex', 256)

timedelta64 = CType('timedelta', 64)
datetime64 = CType('datetime', 64)

ulonglong  = CType('ulonglong')

byte = int8
short = int16
int_ = int32
longlong = int64

ubyte = uint8
ushort = uint16
ulonglong = uint64

half = float16
float_ = float32
double = float64
longdouble = float128

single = float32
longfloat = float128

void = CType('void')
object_ = pyobj = CType('object')

na = Null
top = Top()
dynamic = Dynamic()
NullRecord = Record(())
blob = Blob()

string = String

Stream = Range(IntegerConstant(0), None)

Type.register('int', int_)
Type.register('float', float_)

Type.register('NA', Null)
Type.register('Stream', Stream)
Type.register('?', Dynamic)
Type.register('top', top)
Type.register('blob', blob)

Type.register('string', String())
Type.register('string8', String(8))
Type.register('string16', String(16))
Type.register('string24', String(24))
Type.register('string32', String(32))
Type.register('string64', String(64))
Type.register('string128', String(128))
Type.register('string256', String(256))

#------------------------------------------------------------------------
# Deconstructors
#------------------------------------------------------------------------

#  Dimensions
#      |
#  ----------
#  1, 2, 3, 4,  int32
#               -----
#                 |
#              Measure

def extract_dims(ds):
    """ Discard measure information and just return the
    dimensions
    """
    if isinstance(ds, CType):
        return tuple()
    return ds.parameters[:-1]

def extract_measure(ds):
    """ Discard shape information and just return the measure
    """
    if isinstance(ds, CType):
        return ds
    return ds.parameters[-1]

def is_simple(ds):
    # Unit Type
    if not ds.composite:
        if isinstance(ds, (Fixed, IntegerConstant, CType)):
            return True

    # Composite Type
    else:
        for dim in ds:
            if not isinstance(dim, (Fixed, IntegerConstant, CType)):
                return False
        return True

def promote_cvals(*vals):
    """
    Promote Python values into the most general dshape containing
    all of them. Only defined over simple CType instances.

    >>> promote_vals(1,2.)
    dshape("float64")
    >>> promote_vals(1,2,3j)
    dshape("complex128")
    """

    promoted = np.result_type(*vals)
    datashape = CType.from_numpy_dtype(promoted)
    return datashape

#------------------------------------------------------------------------
# Utility Functions
#------------------------------------------------------------------------

def cat_dshapes(dslist):
    """
    Concatenates a list of dshapes together along
    the first axis. Raises an error if there is
    a mismatch along another axis or the measures
    are different.

    Requires that the leading dimension be a known
    size for all data shapes.
    TODO: Relax this restriction to support
          streaming dimensions.
    """
    if len(dslist) == 0:
        raise ValueError('Cannot concatenate an empty list of dshapes')
    elif len(dslist) == 1:
        return dslist[0]

    outer_dim_size = operator.index(dslist[0][0])
    inner_ds = dslist[0][1:]
    for ds in dslist[1:]:
        outer_dim_size += operator.index(ds[0])
        if ds[1:] != inner_ds:
            raise ValueError(('The datashapes to concatenate must all match after'
                            ' the first dimension (%s vs %s)') %
                            (inner_ds, ds[1:]))
    return DataShape([Fixed(outer_dim_size)] + list(inner_ds))


#------------------------------------------------------------------------
# Python Compatibility
#------------------------------------------------------------------------

def from_python_scalar(scalar):
    """
    Return a datashape ctype for a python scalar.
    """
    if isinstance(scalar, int):
        return int_
    elif isinstance(scalar, float):
        return double
    elif isinstance(scalar, complex):
        return complex128
    elif isinstance(scalar, (str, unicode)):
        return string
    elif isinstance(scalar, datetime.timedelta):
        return timedelta64
    elif isinstance(scalar, datetime.datetime):
        return datetime64
    else:
        return pyobj

#------------------------------------------------------------------------
# NumPy Compatibility
#------------------------------------------------------------------------

class NotNumpyCompatible(Exception):
    """
    Raised when we try to convert a datashape into a NumPy dtype
    but it cannot be ceorced.
    """
    pass

def to_dtype(ds):
    """ Throw away the shape information and just return the
    measure as NumPy dtype instance."""
    return to_numpy(extract_measure(ds))

def to_numpy(ds):
    """
    Downcast a datashape object into a Numpy (shape, dtype) tuple if
    possible.

    >>> to_numpy(dshape('5, 5, int32'))
    (5,5), dtype('int32')
    """

    if isinstance(ds, CType):
        return ds.to_dtype()

    # XXX: fix circular deps for DeclMeta
    if hasattr(ds, 'to_dtype'):
        return None, ds.to_dtype()

    shape = tuple()
    dtype = None

    #assert isinstance(ds, DataShape)

    # The datashape dimensions
    for dim in extract_dims(ds):
        if isinstance(dim, IntegerConstant):
            shape += (dim,)
        elif isinstance(dim, Fixed):
            shape += (dim.val,)
        elif isinstance(dim, TypeVar):
            shape += (-1,)
        else:
            raise NotNumpyCompatible('Datashape dimension %s is not NumPy-compatible' % dim)

    # The datashape measure
    msr = extract_measure(ds)
    if isinstance(msr, CType):
        dtype = msr.to_dtype()
    elif isinstance(msr, Blob):
        dtype = np.dtype('object')
    elif isinstance(msr, Record):
        dtype = msr.to_dtype()
    else:
        raise NotNumpyCompatible('Datashape measure %s is not NumPy-compatible' % dim)

    if type(dtype) != np.dtype:
        raise NotNumpyCompatible('Internal Error: Failed to produce NumPy dtype')
    return (shape, dtype)


def from_numpy(shape, dt):
    """
    Upcast a (shape, dtype) tuple if possible.

    >>> from_numpy((5,5), dtype('int32'))
    dshape('5, 5, int32')
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
        return DataShape(parameters=(map(Fixed, shape)+[measure]))

def from_char(c):
    dtype = np.typeDict[c]
    return from_numpy((), np.dtype(dtype))

def from_dtype(dt):
    return from_numpy((), dt)

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
# Argument Munging
#------------------------------------------------------------------------

def flatten(it):
    for a in it:
        if a.composite:
            for b in iter(a):
                yield b
        else:
            yield a

def table_like(ds):
    return type(ds[-1]) is Record

def array_like(ds):
    return not table_like(ds)

def _reduce(x):
    if isinstance(x, Record):
        return [(k, _reduce(v)) for k,v in x.parameters[0]]
    elif isinstance(x, DataShape):
        return map(_reduce, x.parameters)
    elif isinstance(x, TypeVar):
        import pdb; pdb.set_trace()
    else:
        return x
