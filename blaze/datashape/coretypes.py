"""
This defines the DataShape type system.
"""

import numpy as np

import datetime
from numbers import Integral

try:
    from numba.minivect import minitypes
    have_minivect = True
except ImportError:
    have_minivect = False

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

class Primitive(object):
    composite = False
    __metaclass__ = Type

    def __init__(self, *parameters):
        self.parameters = parameters

    def __rmul__(self, other):
        if not isinstance(other, (DataShape, Primitive)):
            other = shape_coerce(other)
        return product(other, self)

    def __mul__(self, other):
        if not isinstance(other, (DataShape, Primitive)):
            other = shape_coerce(other)
        return product(other, self)

class Null(Primitive):
    """
    The null datashape.
    """
    def __str__(self):
        return expr_string('null', None)

class Integer(Primitive):
    """
    Integers at the level of constructor it just means integer in the
    sense of of just an integer value to a constructor.

    ::

        1, int32   # 1 is Fixed
        Range(1,5) # 1 is Integer

    """

    def __init__(self, i):
        assert isinstance(i, int)
        self.val = i

    def __eq__(self, other):
        if type(other) is Integer:
            return self.val == other.val
        else:
            return False

    def __str__(self):
        return str(self.val)

class Dynamic(Primitive):
    """
    The dynamic type allows an explicit upcast and downcast from any
    type to ``?``.
    """

    def __str__(self):
        return '?'

    def __repr__(self):
        # emulate numpy
        return ''.join(["dshape(\"", str(self), "\")"])

class Top(Primitive):
    """ The top type """

    def __str__(self):
        return 'top'

    def __repr__(self):
        # emulate numpy
        return ''.join(["dshape(\"", str(self), "\")"])

class Blob(Primitive):
    """ Blob type, large variable length string """

    def __str__(self):
        return 'blob'

    def __repr__(self):
        # emulate numpy
        return ''.join(["dshape(\"", str(self), "\")"])

class Varchar(Primitive):
    """ Blob type, small variable length string """


    def __init__(self, maxlen):
        assert isinstance(maxlen, Integer)
        self.maxlen = maxlen.val

    def __str__(self):
        return 'varchar(maxlen=%i)' % self.maxlen

    def __repr__(self):
        return expr_string('varchar', [self.maxlen])

class String(Primitive):
    """ Fixed length string container """

    def __init__(self, fixlen):
        if isinstance(fixlen, int):
            self.fixlen = fixlen
        elif isinstance(fixlen, Integer):
            self.fixlen = fixlen.val
        else:
            raise ValueError()

    def __str__(self):
        return 'string(%i)' % self.fixlen

    def __repr__(self):
        return expr_string('string', [self.fixlen])

#------------------------------------------------------------------------
# Base Types
#------------------------------------------------------------------------

# TODO: figure out consistent spelling for this
#
#   - DataShape
#   - Datashape

class DataShape(object):
    """ The Datashape class, implementation for generic composite
    datashape objects """

    __metaclass__ = Type
    composite = False

    def __init__(self, parameters=None, name=None):

        if type(parameters) is DataShape:
            self.paramaeters = parameters

        elif len(parameters) > 0:
            self.parameters = tuple(flatten(parameters))
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

    # TODO these are kind of hackish, remove
    def __rmul__(self, other):
        if not isinstance(other, (DataShape, Primitive)):
            other = shape_coerce(other)
        return product(other, self)

    def __mul__(self, other):
        if not isinstance(other, (DataShape, Primitive)):
            other = shape_coerce(other)
        return product(other, self)

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
            return False
        else:
            raise TypeError('Cannot compare non-datashape to datashape')

    def __repr__(self):
        # need double quotes to form valid aterm, also valid
        # Python
        return ''.join(["dshape(\"", str(self), "\")"])

    @property
    def shape(self):
        return self.parameters[:-1]

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

class CType(DataShape):
    """
    Symbol for a sized type mapping uniquely to a native type.
    """
    cls = MEASURE

    def __init__(self, ctype, size=None):
        if size:
            assert 1 <= size < (2**23-1)
            label = ctype + str(size)
            self.parameters = [label]
            self.name = label
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
    def from_dtype(self, dt):
        """
        From Numpy dtype.

        >>> CType.from_dtype(dtype('int32'))
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

    @property
    def byeteorder(self):
        raise NotImplementedError()

#------------------------------------------------------------------------
# Fixed
#------------------------------------------------------------------------

class Fixed(Atom):
    """
    Fixed dimension.
    """

    def __init__(self, i):
        assert isinstance(i, Integral)
        self.val = i
        self.parameters = [self.val]

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
    A free variable in the dimension specifier. Not user facing.
    """

    def __init__(self, symbol):
        self.symbol = symbol

    def __str__(self):
        return str(self.symbol)

    def __eq__(self, other):
        if isinstance(other, TypeVar):
            return self.symbol == other.symbol
        else:
            return False

class Range(Atom):
    """
    Range type representing a bound or unbound interval of
    of possible Fixed dimensions.
    """

    def __init__(self, a, b=False):
        if type(a) is int:
            self.a = a
        else:
            self.a = a.val

        if type(b) is int:
            self.b = b
        elif b is False or b is None:
            self.b = b
        else:
            self.b = b.val

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

    def __eq__(self, other):
        return False

class Option(Atom):
    """
    A sum type for nullable measures unit types. Can be written
    as a tagged union with with ``left`` as ``null`` and
    ``right`` as a measure.
    """

    def __init__(self, ty):
        self.parameters = [ty]

class Enum(Atom):
    """
    A finite enumeration of Fixed dimensions.
    """

    def __str__(self):
        # Use c-style enumeration syntax
        return expr_string('', self.parameters, '{}')

    def __eq__(self, other):
        return False

    def __getitem__(self, index):
        return self.parameters[index]

    def __len__(self):
        return len(self.parameters)

class Union(Atom):
    """
    A untagged union is a datashape for a value that may hold
    several but fixed datashapes.
    """

    def __str__(self):
        return expr_string('', self.parameters, '{}')

    def __getitem__(self, index):
        return self.parameters[index]

    def __len__(self):
        return len(self.parameters)

class Record(DataShape):
    """
    A composite data structure of ordered fields mapped to types.
    """

    def __init__(self, fields):
        # This is passed in with a OrderedDict so field order is
        # preserved. Using RecordDecl there is some magic to also
        # ensure that the fields align in the order they are
        # declared.
        self.__d = dict(fields)
        self.__k = [f[0] for f in fields]
        self.__v = [f[1] for f in fields]

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
        return str(self)

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

int_       = CType('int')
long_      = CType('long')
bool_      = CType('bool')
float_     = CType('float')
double     = CType('double')
short      = CType('short')
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
longdouble = float128

void = CType('void')
object_ = pyobj = CType('object')

na = Null
top = Top()
dynamic = Dynamic()
NullRecord = Record(())

blob = Blob()
string = String

Stream = Range(Integer(0), None)

Type.register('NA', Null)
Type.register('Stream', Stream)
Type.register('?', Dynamic)
Type.register('top', top)
Type.register('blob', blob)
Type.register('string8', String(8))
Type.register('string16', String(16))
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
        raise Exception("No Dimensions")
    return ds.parameters[0:-2]

def extract_measure(ds):
    """ Discard shape information and just return the measure
    """
    if isinstance(ds, CType):
        return ds
    return ds.parameters[-1]

def is_simple(ds):
    # Unit Type
    if not ds.composite:
        if isinstance(ds, (Fixed, Integer, CType)):
            return True

    # Composite Type
    else:
        for dim in ds:
            if not isinstance(dim, (Fixed, Integer, CType)):
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
    datashape = CType.from_dtype(promoted)
    return datashape

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
# Minivect Compatibility
#------------------------------------------------------------------------

def to_minitype(ds):
    # To minitype through NumPy. Discards dimension information.
    return minitypes.map_dtype(to_numpy(extract_measure(ds)))

def to_minivect(ds):
    raise NotImplementedError
    #return (shape, minitype)

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
    for dim in ds:
        if isinstance(dim, Integer):
            shape += (dim,)
        elif isinstance(dim, Fixed):
            shape += (dim.val,)
        elif isinstance(dim, CType):
            dtype = dim.to_dtype()
        elif isinstance(dim, Record):
            dtype = dim.to_dtype()
        else:
            raise NotNumpyCompatible()

    if len(shape) < 0 or type(dtype) != np.dtype:
        raise NotNumpyCompatible()
    return (shape, dtype)


def from_numpy(shape, dt):
    """
    Upcast a (shape, dtype) tuple if possible.

    >>> from_numpy((5,5), dtype('int32'))
    dshape('5, 5, int32')
    """
    dtype = np.dtype(dt)

    if shape == ():
        dimensions = []
    else:
        dimensions = map(Fixed, shape)

    if dtype.fields:
        # Convert the record into a dict of keys to CType
        rec = [(a,CType.from_dtype(b[0])) for a,b in dtype.fields.items()]
        measure = Record(rec)
    else:
        measure = CType.from_dtype(dtype)

    return reduce(product, dimensions + [measure])

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

def doublequote(s):
    if '"' not in s:
        return '"%s"' % s
    else:
        return s

def shape_coerce(ob):
    if type(ob) is int:
        return Integer(ob)
    else:
        raise NotImplementedError()

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
