from collections import namedtuple

#------------------------------------------------------------------------
# Nodes
#------------------------------------------------------------------------

mod = namedtuple('module', 'name body')
imprt = namedtuple('imprt', 'qualified name')
include = namedtuple('include', 'name')
typeset = namedtuple('typeset', 'name types')

# module level definitions
tclass = namedtuple('tclass', 'name params body')
instance = namedtuple('instance', 'name params body meta')

# operator definition pair
opdef = namedtuple('opdef', 'op sig')
opimp = namedtuple('opimp', 'op node')
opalias = namedtuple('opalias', 'op ref')

# function definition pair
fndef = namedtuple('fndef', 'name sig')
fnimp = namedtuple('fnimp', 'name node')

# associated types
eldef = namedtuple('eldef', 'name ty')
tydef = namedtuple('eldef', 'name')

# sort definition
sortdef = namedtuple('sortdef', 'sorts')

# foreign defs
foreign = namedtuple('foreign', 'name call fname sig')

# function type
fn = namedtuple('fn', 'dom cod')
fn.__repr__ = lambda s: '%s -> %s' % (s.dom, s.cod)

# parameterized type
pt = namedtuple('pt', 'con args')
pt.__repr__ = lambda s: '%s %s' % (s.con, ' '.join(str(a) for a in s.args))

# polymorphic typevar
pv = namedtuple('pv', 'id')
pv.__repr__ = lambda s: '\'%s' % s.id

# signature
sig = namedtuple('sig', 'dom cod')
sig.__repr__ = lambda s: '%s -> %s' % (s.dom, s.cod)

tenum = namedtuple('tenum', 'var typeset')
