from collections import namedtuple

#------------------------------------------------------------------------
# Types
#------------------------------------------------------------------------

Type = namedtuple('Type', 'name, kind, zero, binary_ops, unary_ops, cmp_ops, fields')
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
    fields     = {}
)

float_type = Type(
    name       = "float",
    kind       = Unit,
    zero       = 0.0,
    binary_ops = set(['+','-','*','/']),
    unary_ops  = set(['+','-']),
    cmp_ops    = set(['<','<=','>','>=','==','!=']),
    fields     = {}
)

string_type = Type(
    name       = "str",
    kind       = Unit,
    zero       = "",
    binary_ops = set(['+']),
    unary_ops  = set(),
    cmp_ops    = set(['<','<=','>','>=','==','!=']),
    fields     = {}
)

bool_type = Type(
    name       = "bool",
    kind       = Unit,
    zero       = False,
    binary_ops = set(),
    unary_ops  = set(['!']),
    cmp_ops    = set(['==','!=','&&','||']),
    fields     = {}
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
        'data' : (0, int_type),
        'nd'   : (1, int_type),
    }
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
    fields     = {}
)

any_type = Type(
    name       = "any",
    zero       = None,
    kind       = Unit,
    binary_ops = set(),
    unary_ops  = set(),
    cmp_ops    = set(),
    fields     = {}
)

undefined = Type(
    name       = "undefined",
    zero       = None,
    kind       = Exception,
    binary_ops = set(),
    unary_ops  = set(),
    cmp_ops    = set(),
    fields     = {}
)

__all__ = [
    'int_type',
    'float_type',
    'string_type',
    'bool_type',
    'array_type',
    'void_type',
    'any_type',
]
