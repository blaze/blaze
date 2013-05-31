from collections import namedtuple

#------------------------------------------------------------------------
# Types
#------------------------------------------------------------------------

Type = namedtuple('Type',
    'name, kind, zero, binary_ops,\
    unary_ops, cmp_ops, fields, order'
)
Type.__repr__ = lambda s: s.name

TParam = namedtuple('TParam', 'cons, arg')
TArrow = namedtuple('TArrow', 'dom, cod')
TVar   = str

#------------------------------------------------------------------------
# Kinds
#------------------------------------------------------------------------

Kind   = namedtuple('Kind', 'ty')
KArrow = namedtuple('KArrow', 'left, right')

Star  = Kind(None)

Unit  = Star
Param = KArrow(Star, Star)

#------------------------------------------------------------------------
# Scalars
#------------------------------------------------------------------------

int_type = Type(
    name       = "int",
    kind       = Unit,
    zero       = 0,
    binary_ops = set(['+','-','*','/','&','|','^','>>','<<']),
    unary_ops  = set(['+','-','~']),
    cmp_ops    = set(['<','<=','>','>=','==','!=']),
    fields     = {},
    order      = None,
)

float_type = Type(
    name       = "float",
    kind       = Unit,
    zero       = 0.0,
    binary_ops = set(['+','-','*','/']),
    unary_ops  = set(['+','-']),
    cmp_ops    = set(['<','<=','>','>=','==','!=']),
    fields     = {},
    order      = None,
)

string_type = Type(
    name       = "str",
    kind       = Unit,
    zero       = "",
    binary_ops = set(['+']),
    unary_ops  = set(),
    cmp_ops    = set(['<','<=','>','>=','==','!=']),
    fields     = {},
    order      = None,
)

bool_type = Type(
    name       = "bool",
    kind       = Unit,
    zero       = False,
    binary_ops = set(),
    unary_ops  = set(['!']),
    cmp_ops    = set(['==','!=','&&','||']),
    fields     = {},
    order      = None,
)

#------------------------------------------------------------------------
# Aggregate Types
#------------------------------------------------------------------------

array_type = Type(
    name       = "array",
    kind       = Param,
    zero       = None,
    binary_ops = set(),
    unary_ops  = { },
    cmp_ops    = { },
    fields = {
        'nd'   : (1, int_type),
    },
    order      = None,
)

#------------------------------------------------------------------------
# Polymorphic Arrays
#------------------------------------------------------------------------

# Parameters are indicted with $0 and $1 to indicate the first
# and second parameters of the type instance.

DimInfo = Type(
    'DimInfo',
    kind       = Param,
    zero       = None,
    binary_ops = set(),
    unary_ops  = { },
    cmp_ops    = { },
    fields = {
        'dim'     : (0, int_type),
        'strides' : (1, int_type),
    },
    order      = None,
)

Array_C = Type(
    name       = "Array_C",
    kind       = Param,
    zero       = None,
    binary_ops = set(),
    unary_ops  = { },
    cmp_ops    = { },
    fields = {
        'data'  : (0, '$0'),
        'shape' : (1, array_type),
    },
    order      = 'C',
)

Array_F = Type(
    name       = "Array_F",
    kind       = Param,
    zero       = None,
    binary_ops = set(),
    unary_ops  = { },
    cmp_ops    = { },
    fields = {
        'data'  : (0, '$0'),
        'shape' : (1, (array_type, '$0')),
    },
    order      = 'F',
)

Array_S = Type(
    name       = "Array_S",
    kind       = Param,
    zero       = None,
    binary_ops = set(),
    unary_ops  = { },
    cmp_ops    = { },
    fields = {
        'data'  : (0, '$0'),
        'shape' : (1, (array_type, '$0')),
    },
    order      = 'S',
)

#------------------------------------------------------------------------
# Special Types
#------------------------------------------------------------------------

void_type = Type(
    name       = "void",
    zero       = None,
    kind       = Unit,
    binary_ops = set(),
    unary_ops  = set(),
    cmp_ops    = set(),
    fields     = {},
    order      = None,
)

any_type = Type(
    name       = "any",
    zero       = None,
    kind       = Unit,
    binary_ops = set(),
    unary_ops  = set(),
    cmp_ops    = set(),
    fields     = {},
    order      = None,
)

undefined = Type(
    name       = "undefined",
    zero       = None,
    kind       = Exception,
    binary_ops = set(),
    unary_ops  = set(),
    cmp_ops    = set(),
    fields     = {},
    order      = None,
)

__all__ = [
    'int_type',
    'float_type',
    'string_type',
    'bool_type',
    'array_type',
    'void_type',
    'any_type',
    'Array_C',
    'Array_F',
]
