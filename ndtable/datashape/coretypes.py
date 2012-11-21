"""
This defines the DataShape type system.
"""

# TODO: preferably not depend on these
import numpy as np
from numpy import dtype

from struct import calcsize
from string import letters
from itertools import count
from platform import architecture
from numbers import Integral
from operator import methodcaller
from collections import Mapping, Sequence
from utils import ReverseLookupDict

#------------------------------------------------------------------------
# Free Variables
#------------------------------------------------------------------------

free_vars = methodcaller('free')

def _var_generator(prefix=None):
    """
    Generate a stream of unique free variables.
    """
    for a in count(0):
        for b in letters:
            if a == 0:
                yield (prefix or '') + b
            else:
                yield (prefix or '') + ''.join([str(a),str(b)])

var_generator = _var_generator()

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

#------------------------------------------------------------------------
# Argument Munging
#------------------------------------------------------------------------

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

#------------------------------------------------------------------------
# Type Metaclass
#------------------------------------------------------------------------

class Type(type):
    _registry = {}

    __init__ = NotImplemented

    def __new__(meta, name, bases, dct):
        cls = type(name, bases, dct)

        # Don't register abstract classes
        if not dct.get('abstract'):
            Type._registry[name] = cls
            return cls

    @staticmethod
    def register(name, cls):
        assert name not in Type._registry
        Type._registry[name] = cls

# ==================================================================
# Primitives
# ==================================================================

class Primitive(object):
    composite = False

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
    Type agnostic missing value.
    """

    def __str__(self):
        return expr_string('NA', None)

class Integer(Primitive):
    """
    Integers, at the top level this means a Fixed dimension, at
    level of constructor it just means integer in the sense of
    of just an integer value.
    """

    def __init__(self, i):
        assert isinstance(i, Integral)
        self.val = i

    def free(self):
        return set([])

    def __eq__(self, other):
        if type(other) is Integer:
            return self.val == other.val
        else:
            return False

    def __str__(self):
        return str(self.val)

class Dynamic(object):
    """
    The dynamic type allows an explicit upcast and downcast from any
    type to ``?``. This is normally not allowed in most static type
    systems.
    """
    __metaclass__ = Type

    def up(self, ty):
        pass

    def down(self, ty):
        pass

    def __str__(self):
        return '?'

    def __repr__(self):
        return str(self)

Any = Dynamic

# ==================================================================
# Base Types
# ==================================================================

class DataShape(object):
    __metaclass__ = Type

    composite = False
    name = False

    def __init__(self, operands=None, name=None):

        if type(operands) is DataShape:
            self.operands = operands

        elif len(operands) > 0:
            self.operands = tuple(flatten(operands))
            self.composite = True
        else:
            self.operands = tuple()
            self.composite = False

        if name:
            self.name = name
            self.__metaclass__._registry[name] = self

    def size(self):
        """
        In NumPy, size would be a integer value. In Blaze the
        size is now a symbolic object

        A Numpy array of size (2,3) has size
            np.prod([2,3]) = 6
        A NDTable of datashape (a,b,2,3,int32) has size
            6*a*b
        """
        pass

    def __getitem__(self, index):
        return self.operands[index]

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
            return ' '.join(map(str, self.operands))

    def _equal(self, other):
        return all(a==b for a,b in zip(self, other))

    def __eq__(self, other):
        if type(other) is DataShape:
            return False
        else:
            raise TypeError('Cannot non datashape to datashape')

    def __repr__(self):
        return str(self)

class Atom(DataShape):
    """
    Atoms for arguments to constructors of types, not types in
    and of themselves. Parser artifacts, if you like.
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
# Native Types
#------------------------------------------------------------------------

class CType(DataShape):
    """
    Symbol for a sized type mapping uniquely to a native type.
    """

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
        return Type._registry(dt.name)

    def size(self):
        # TODO: no cheating!
        return dtype(self.name).itemsize

    def to_struct(self):
        """
        To struct code.
        """
        return dtype(self.name).char


    def to_dtype(self):
        """
        To Numpy dtype.
        """
        return dtype(self.name)

    def to_minitype(self):
        """
        To minivect AST node ( through Numpy for now ).
        """
        from ndtable.engine.mv import minitypes
        return minitypes.map_dtype(dtype(self.name))

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
        self.operands = [self.val]

    def __eq__(self, other):
        if type(other) is Fixed:
            return self.val == other.val
        else:
            return False

    def __str__(self):
        return str(self.val)

#------------------------------------------------------------------------
# Variable
#------------------------------------------------------------------------

class TypeVar(Atom):
    """
    A free variable in the dimension specifier.
    """

    def __init__(self, symbol):
        self.symbol = symbol

    def free(self):
        return set([self.symbol])

    def __str__(self):
        return str(self.symbol)

    def __eq__(self, other):
        if isinstance(other, TypeVar):
            return self.symbol == other.symbol
        else:
            return False

# Internal-like range of dimensions, the special case of
# [0, inf) is aliased to the type Stream.
class Var(Atom):
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
        return expr_string('Var', [self.lower, self.upper])

#------------------------------------------------------------------------
# Derived Dimensions
#------------------------------------------------------------------------

class Bitfield(Atom):

    def __init__(self, size):
        self.size = size.val
        self.parameters = [size]

#------------------------------------------------------------------------
# Aggregate
#------------------------------------------------------------------------

class Either(Atom):
    """
    Taged union with two slots.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.parameters = [a,b]

class Enum(Atom, Sequence):
    """
    A finite enumeration of Fixed dimensions that a datashape is over,
    in order.
    """

    def __str__(self):
        # Use c-style enumeration syntax
        return expr_string('', self.parameters, '{}')

    def __getitem__(self, index):
        return self.parameters[index]

    def __len__(self):
        return len(self.parameters)

class Union(Atom, Sequence):
    """
    A union of possible datashapes that may occupy the same
    position.
    """

    def __str__(self):
        return expr_string('', self.parameters, '{}')

    def __getitem__(self, index):
        return self.parameters[index]

    def __len__(self):
        return len(self.parameters)

class Record(DataShape, Mapping):
    """
    A composite data structure with fields mapped to types.
    """

    def __init__(self, **kwargs):
        self.d = kwargs
        self.k = kwargs.keys()
        self.v = kwargs.values()

    @property
    def fields(self):
        return self.d

    @property
    def names(self):
        return self.k

    def free(self):
        return set(self.k)

    def __eq__(self, other):
        if isinstance(other, Record):
            return self.d == other.d
        else:
            return False

    def __call__(self, key):
        return self.d[key]

    def __iter__(self):
        return zip(self.k, self.v)

    def __len__(self):
        return len(self.k)

    def __str__(self):
        return 'Record ( ' + ''.join([('%s = %s, ' % (k,v)) for k,v in zip(self.k, self.v)]) + ')'

    def __repr__(self):
        return 'Record ' + repr(self.d)

# Memory Types
# ============

class RemoteSpace(object):
    def __init__(self, blaze_uri):
        pass

class LocalMemory(object):
    def __init__(self):
        self.bounds = ( 0x0, 0xffffffffffffffff )

class SharedMemory(object):
    def __init__(self, key):
        # something like:
        # shmid = shmget(key)
        # shmat(shmid, NULL, 0)
        self.bounds = ( 0x0, 0x1 )

class Ptr(Atom):
    """

    Usage:
    ------
    Pointer to a integer in local memory::

        *int32

    Pointer to a 4x4 matrix of integers in local memory::

        *(4, 4, int32)

    Pointer to a record in local memory::

        *{x: int32, y:int32, label: string}

    Pointer to integer in a shared memory segement keyed by 'foo'::

        *(int32 (shm 'foo'))

    Pointer to integer on a array server 'bar'::

        *(int32 (rmt array://bar))

    """

    def __init__(self, pointee, addrspace=None):
        self.pointee = pointee

        if addrspace:
            self.addrspace = addrspace
            self.parameters = [pointee, addrspace]
        else:
            self.parameters = [pointee]
            self.addrspace = LocalMemory()

    @property
    def byte_bounds(self):
        return self.addrspace.bounds

    @property
    def local(self):
        return isinstance(self.addrspace, LocalMemory())

    @property
    def remote(self):
        return isinstance(self.addrspace, RemoteSpace())

# Class derived Records
# =====================
# They're just records but can be constructed like Django models.

def derived(sig):
    from parse import parse
    sig = parse(sig)
    def a(fn):
        return sig
    return a

# Constructions
# =============

# Right now we only have one operator (,) which constructs
# product types ( ie A * B ). We call these dimensions.

# It is neccesarry that if forall z = x * y then
#   fst(z) * snd(z) = z

# product :: A -> B -> A * B
def product(A,B):
    if A.composite and B.composite:
        f = A.operands
        g = B.operands

    elif A.composite:
        f = A.operands
        g = (B,)

    elif B.composite:
        f = (A,)
        g = B.operands

    else:
        f = (A,)
        g = (B,)

    return DataShape(operands=(f+g))

# fst :: A * B -> A
def fst(ds):
    return ds[0]

# snd :: A * B -> B
def snd(ds):
    return ds[1:]

# Coproduct is the dual to the (,) operator. It constructs sum
# types ( ie A + B ).
def coprod(A, B):
    return Either(A,B)

# left  :: A + B -> A
def left(ds):
    return ds.parameters[0]

# right :: A + B -> B
def right(ds):
    return ds.parameters[1]

# Machines Types
# ==============
# At the type level these are all singleton types, they take no
# arguments in their constructors.

plat = calcsize('@P') * 8

long_      = CType('long')
bool_      = CType('bool')
double     = CType('double')
short      = CType('short')
longdouble = CType('longdbouble')
char       = CType('char')

if plat == 32:
    int8  = CType('int', 8)
    int16 = CType('int', 16)
    int32 = CType('int', 32)
    int64 = CType('int', 64)
    int_  = int32

elif plat == 64:
    int8  = CType('int', 8)
    int16 = CType('int', 16)
    int32 = CType('int', 32)
    int64 = CType('int', 64)
    int_  = int64

uint       = CType('uint')
ulong      = CType('ulong')
ulonglong  = CType('ulonglong')


uint8      = CType('uint',  8)
uint16     = CType('uint', 16)
uint32     = CType('uint', 32)
uint64     = CType('uint', 64)

if plat == 32:
    float8       = CType('float', 8)
    float16      = CType('float', 16)
    float32      = CType('float', 32)
    float64      = CType('float', 64)
    float128     = CType('float', 128)
    float_       = float32

elif plat == 64:
    float8       = CType('float', 8)
    float16      = CType('float', 16)
    float32      = CType('float', 32)
    float64      = CType('float', 64)
    float128     = CType('float', 128)
    float_       = float64

complex64  = CType('complex' , 64)
complex128 = CType('complex', 128)
complex256 = CType('complex', 256)

void       = CType('void')
pyobj      = CType('object')

# TODO: differentiate between fixed-length and variable-length
# strings once we figure out how to implement this!
string     = CType('string')

Stream = Var(Integer(0), None)

# The null record
NullRecord = Record()

na = Null
top = Any()

Type.register('NA', Null)
Type.register('Stream', Stream)

#------------------------------------------------------------------------
# Shorthand
#------------------------------------------------------------------------

O = pyobj
b1 = bool_

i1 = int8
i2 = int16
i4 = int32
i8 = int64

u1 = uint8
u2 = uint16
u4 = uint32
u8 = uint64

f1 = float8
f2 = float16
f4 = float32
f8 = float64
f16 = float128

f = float_
d = double

c8  = complex64
c16 = complex128
c32 = complex256

S = string

#------------------------------------------------------------------------
# Numpy Compatability
#------------------------------------------------------------------------


# TODO: numpy structured arrays
def to_numpy(ds):
    """
    Downcast a datashape object into a Numpy (shape, dtype) tuple if
    possible.

    >>> to_numpy(dshape('5, 5, int32'))
    (5,5), dtype('int32')
    """

    shape = tuple()
    dtype = None

    assert isinstance(ds, DataShape)
    for dim in ds:
        if isinstance(dim, Integer):
            shape += (dim,)
        if isinstance(dim, CType):
            dtype += (dim,)

    assert len(shape) > 0 and dtype, "Could not convert"
    return (shape, dtype)


def from_numpy(shape, dt):
    """
    Upconvert a datashape object into a Numpy
    (shape, dtype) tuple if possible.
    i.e.
      5,5,in32 -> ( (5,5), dtype('int32') )
    """

    dimensions = map(Fixed, shape)
    measure = CType.from_dtype(dt)

    return reduce(product, dimensions + [measure])

def table_like(ds):
    return type(ds[-1]) is Record

def array_like(ds):
    return not table_like(ds)

def expand(ds):
    """
    Expand the datashape into a tree like structure of nested
    structure.
    """
    x = ds[0]

    #       o
    #      /|\
    #  1 o ... o n

    if isinstance(x, Fixed):
        y = list(expand(ds[1:]))
        for a in xrange(0, x.val):
            yield y

    elif isinstance(x, Enum):
        y = list(expand(ds[1:]))
        for a in x.parameters:
            for b in a:
                yield y

    #       o
    #       |
    #       o
    #       |
    #       o

    elif isinstance(x, Record):
        for a in x.k:
            yield a

    else:
        yield x
