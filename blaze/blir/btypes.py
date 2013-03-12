#------------------------------------------------------------------------
# Kinds
#------------------------------------------------------------------------

class Kind(object):
    def __init__(self, ty=None):
        self.ty = ty
    def __repr__(self):
        if self.ty == None:
            return '*'
        else:
            return repr(self.ty)

class KArrow(Kind):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return '%s -> %s' % (self.left, self.right)

Star = Kind(None)
# <kind> :: *
Unit = Star
# <kind> :: * -> *
Param = KArrow(Star, Star)

#------------------------------------------------------------------------
# Types
#------------------------------------------------------------------------

class Type(object):

    def __init__(self, name, kind, zero, binary_ops,unary_ops,cmp_ops):
        self.name = name
        self.zero = zero
        self.kind = kind
        self.binary_ops = binary_ops
        self.unary_ops = unary_ops
        self.cmp_ops = cmp_ops

    def __eq__(self, other):
        if isinstance(other, Type):
            return self.name == other.name
        else:
            return False

    def __repr__(self):
        return '<%s>' % self.name

class PType(Type):
    def __init__(self, cons, arg):
        self.name = 'array'
        self.kind = Param
        self.cons = cons
        self.arg = arg

    def __eq__(self, other):
        if isinstance(other, PType):
            return self.cons.name == other.cons.name
        else:
            return False

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return '<%s[%s]>' % (self.cons.name, self.arg.name)

#------------------------------------------------------------------------
# Scalars
#------------------------------------------------------------------------

int_type = Type(
    name       = "int",
    kind       = Unit,
    zero       = 0,
    binary_ops = { '+', '-', '*', '/' },
    unary_ops  = { '+', '-' },
    cmp_ops    = { '<', '<=', '>', '>=', '==', '!=' }
)

float_type = Type(
    name       = "float",
    zero       = 0.0,
    kind       = Unit,
    binary_ops = { '+', '-', '*', '/' },
    unary_ops  = { '+', '-' },
    cmp_ops    = { '<', '<=', '>', '>=', '==', '!=' }
)

string_type = Type(
    name       = "str",
    zero       = "",
    kind       = Unit,
    binary_ops = { '+' },
    unary_ops  = set(),
    cmp_ops    = { '<', '<=', '>', '>=', '==', '!=' }
)

bool_type = Type(
    name       = "bool",
    zero       = False,
    kind       = Unit,
    binary_ops = set(),
    unary_ops  = { '!' },
    cmp_ops    = { '==', '!=', '&&', '||' }
)

#------------------------------------------------------------------------
# Aggregate Types
#------------------------------------------------------------------------

array_type = Type(
    name       = "array",
    zero       = None,
    kind       = Param,
    binary_ops = set(),
    unary_ops  = { },
    cmp_ops    = { }
)

blaze_type = Type(
    name       = "blaze",
    zero       = None,
    kind       = Param, # parameterized by datashape
    binary_ops = { },
    unary_ops  = { },
    cmp_ops    = { }
)

#------------------------------------------------------------------------
# Special Types
#------------------------------------------------------------------------

void_type = Type(
    name       = "void",
    zero       = None,
    kind       = Unit,
    binary_ops = set(),
    unary_ops  = { },
    cmp_ops    = { }
)

any_type = Type(
    name       = "any",
    zero       = None,
    kind       = Unit,
    binary_ops = set(),
    unary_ops  = { },
    cmp_ops    = { }
)

undefined = Type(
    name       = "undefined",
    zero       = Exception,
    kind       = Exception,
    binary_ops = { },
    unary_ops  = { },
    cmp_ops    = { }
)

builtin_types = [
    int_type,
    float_type,
    string_type,
    bool_type,
    array_type,
    blaze_type,
    void_type,
    any_type,
]
