"""
AIR types.
"""

from __future__ import print_function, division, absolute_import

from itertools import starmap
from collections import namedtuple
from functools import partial

from .utils import invert, hashable


alltypes = set()


class Type(object):
    """Base of types"""

    def __eq__(self, other):
        incompatible = (type(self) != type(other) and not (self.is_typedef or
                                                           other.is_typedef))
        if not isinstance(other, Type) or incompatible:
            return False

        self_recursive = recursive_terms(self)
        other_recursive = recursive_terms(other)

        if len(self_recursive) != len(other_recursive):
            return False
        elif self_recursive:
            return compare_recursive(self_recursive, other_recursive, {}, self, other)
        else:
            return (super(Type, self).__eq__(other) or
                    (self.is_typedef and self.type == other) or
                    (other.is_typedef and other.type == self))

    def __ne__(self, other):
        return not (self == other)

    def __nonzero__(self):
        return True

    def __hash__(self):
        if self.is_struct:
            return 0 # TODO: better hashing
        obj = tuple(tuple(c) if isinstance(c, list) else c for c in self)
        return hash(obj)


def compare_recursive(rec1, rec2, mapping, t1, t2):
    """Structural comparison of recursive types"""
    cmp = partial(compare_recursive, rec1, rec2, mapping)

    sub1 = subterms(t1)
    sub2 = subterms(t2)

    if id(t1) in rec1:
        if id(t1) in mapping:
            return mapping[id(t1)] == id(t2)

        mapping[id(t1)] = id(t2)

    if bool(sub1) ^ bool(sub2) or type(t1) != type(t2):
        return False
    elif not sub1:
        return t1 == t2 # Unit types
    elif t1.is_struct:
        return (t1.names == t2.names and
                all(starmap(cmp, zip(t1.types, t2.types))))
    elif t1.is_function:
        return (t1.varargs == t2.varargs and
                cmp(t1.restype, t2.restype) and
                all(starmap(cmp, zip(t1.argtypes, t2.argtypes))))
    elif t1.is_vector or t1.is_array:
        return t1.count == t2.count and cmp(t1.base, t2.base)
    elif t1.is_pointer:
        return cmp(t1.base, t2.base)


def subterms(type):
    if type.is_struct:
        return type.types
    elif type.is_pointer or type.is_vector or type.is_array:
        return [type.base]
    elif type.is_function:
        return [type.restype] + list(type.argtypes)
    else:
        return []


def recursive_terms(type, seen=None, recursive=None):
    """Find all recursive terms in a type"""
    if seen is None:
        seen = set()
        recursive = set()

    if id(type) in seen:
        recursive.add(id(type))
        return recursive

    seen.add(id(type))
    for subterm in subterms(type):
        recursive_terms(subterm, seen, recursive)
    seen.remove(id(type))

    return recursive

def typetuple(name, elems):
    def __str__(self):
        from .ir import pretty
        return pretty.ftype(self)

    def __repr__(self):
        return "%s(%s)" % (name, ", ".join(str(getattr(self, attr)) for attr in elems))

    ty = type(name, (Type, namedtuple(name, elems)), {'__str__': __str__,
                                                      '__repr__': __repr__})
    alltypes.add(ty)
    return ty

VoidT      = typetuple('Void',     [])
Boolean    = typetuple('Bool',     [])
Integral   = typetuple('Int',      ['bits', 'unsigned'])
Real       = typetuple('Real',     ['bits'])
Array      = typetuple('Array',    ['base', 'count'])
Vector     = typetuple('Vector',   ['base', 'count'])
Struct     = typetuple('Struct',   ['names', 'types'])
Pointer    = typetuple('Pointer',  ['base'])
Function   = typetuple('Function', ['restype', 'argtypes', 'varargs'])
ExceptionT = typetuple('Exception',[])
BytesT     = typetuple('Bytes',    [])
OpaqueT    = typetuple('Opaque',   []) # Some type we make zero assumptions about

# These are user-defined types
# Complex    = typetuple('Complex',  ['base'])
# ObjectT    = typetuple('Object',   [])

class Typedef(typetuple('Typedef',  ['name', 'type'])):
    def __init__(self, name, ty):
        setattr(self, 'is_' + type(ty).__name__.lower(), True)


for ty in alltypes:
    attr = 'is_' + ty.__name__.lower()
    for ty2 in alltypes:
        setattr(ty2, attr, False)
    setattr(ty, attr, True)

# ______________________________________________________________________
# Types

Void    = VoidT()
Bool    = Boolean()
Int8    = Integral(8,  False)
Int16   = Integral(16, False)
Int32   = Integral(32, False)
Int64   = Integral(64, False)
Int128  = Integral(128, False)
UInt8   = Integral(8,  True)
UInt16  = Integral(16, True)
UInt32  = Integral(32, True)
UInt64  = Integral(64, True)
UInt128 = Integral(128, True)

Vector64x2 = Vector(UInt64, 2)
Vector32x4 = Vector(UInt32, 4)
Vector16x8 = Vector(UInt16, 8)

Float32  = Real(32)
Float64  = Real(64)
# Float128 = Real(128)

# Object    = ObjectT()
Exception = ExceptionT()
Bytes     = BytesT()
Opaque    = OpaqueT()

# Typedefs
Char      = Typedef("Char", Int8)
Short     = Typedef("Short", Int16)
Int       = Typedef("Int", Int32)
Long      = Typedef("Long", Int32)
LongLong  = Typedef("LongLong", Int32)

UChar     = Typedef("UChar", UInt8)
UShort    = Typedef("UShort", UInt16)
UInt      = Typedef("UInt", UInt32)
ULong     = Typedef("ULong", UInt32)
ULongLong = Typedef("ULongLong", UInt32)

# ______________________________________________________________________

signed_set   = frozenset([Int8, Int16, Int32, Int64, Int128])
unsigned_set = frozenset([UInt8, UInt16, UInt32, UInt64, UInt128])
int_set      = signed_set | unsigned_set
float_set    = frozenset([Float32, Float64])
# complex_set  = frozenset([Complex64, Complex128])
bool_set     = frozenset([Bool])
numeric_set  = int_set | float_set # | complex_set
scalar_set   = numeric_set | bool_set

# ______________________________________________________________________
# Internal

VirtualTable  = typetuple('VirtualTable',  ['obj_type'])
VirtualMethod = typetuple('VirtualMethod', ['obj_type'])

# ______________________________________________________________________
# Parsing

def parse_type(s):
    from pykit.parsing import parser
    return parser.build(parser.parse(s, parser.type_parser))

# ______________________________________________________________________
# Typeof

typing_defaults = {
    bool:       Bool,
    int:        Int32,
    float:      Float64,
    # These types are not actually supported
    str:        Bytes,
    bytes:      Bytes,
}

def typeof(value):
    """Python value -> type"""
    return typing_defaults[type(value)]

# ______________________________________________________________________
# Convert

conversion_map = invert(typing_defaults)
conversion_map.update(dict.fromkeys(int_set, int))
conversion_map.update(dict.fromkeys(float_set, float))
# conversion_map.update(dict.fromkeys(complex_set, complex))

def convert(value, dst_type):
    """(python value, type) -> converted python value"""
    if dst_type.is_typedef:
        dst_type = dst_type.type
    converter = conversion_map[dst_type]
    return converter(value)

# ______________________________________________________________________

type2name = dict((v, n) for n, v in globals().items() if hashable(v))
typename = type2name.__getitem__

def resolve_typedef(type):
    while type.is_typedef:
        type = type.type
    return type
