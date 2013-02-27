# -*- coding: utf-8 -*-

from copy import deepcopy
from pprint import pformat
from functools import partial
from collections import defaultdict

import prims
import debug_passes

import qualname
import nodes as N

#------------------------------------------------------------------------
# Sorts
#------------------------------------------------------------------------

pysorts = {
    'int'     : 'int',
    'float'   : 'float',
    'complex' : 'complex',
    'index'   : 'tuple',
    'bool'    : 'bool',
    'any'     : 'object',
}

infix_map = {
    '+'  : '__add__',
    '-'  : '__sub__',
    '*'  : '__mul__',
    '/'  : '__truediv__',
    '%'  : '__mod__',
    '//' : '__floordiv__',
    '**' : '__pow__',
    '&'  : '__and__',
    '|'  : '__or__',
    '^'  : '__xor__',
    '<<' : '__lshift__',
    '>>' : '__rshift__',
    '==' : '__eq__',
    '!=' : '__ne__',
    '<'  : '__lt__',
    '>=' : '__ge__',
    '>'  : '__gt__',
    '<=' : '__le__',
    'in' : '__contains__'
}

#------------------------------------------------------------------------
# Debug Options
#------------------------------------------------------------------------

class opts:
    ddump_parse  = True
    ddump_inst   = False
    ddump_mod    = False
    ddump_alias  = False
    ddump_match  = False
    ddump_adhoc  = False
    ddump_bound  = False
    ddump_classes = False

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def issimple(ty):
    if isinstance(ty, tuple) and len(ty) == 1:
        return True
    return False

def isparametric(ty):
    if isinstance(ty, tuple) and len(ty) > 1:
        return True
    return False

#------------------------------------------------------------------------
# Local resolution
#------------------------------------------------------------------------

# Resolve the abstract type signature in the presense of the
# class definition.

# trait Foo[T x0 y1 y2 ... ]:
#   fun fizz :: <-------------+
#                             |
#                 +-----------+ Resolve the signature context
#                 |           | in the presence of the
#          |-------------|    | implementation
# impl Foo[T y0 y1 y2 ... ]:  |
#   +------------+            |
#   | fun fizz   | -----------+
#   +------------+

def resolve(ctx, ty):

    # parameterized type
    if isinstance(ty, N.pt):
        con = resolve(ctx, ty.con)
        args = map(partial(resolve, ctx), ty.args)
        return N.pt(con, args)

    # function type
    elif isinstance(ty, N.fn):
        dom = resolve(ctx, ty.dom)
        cod = resolve(ctx, ty.cod)
        return N.fn(dom, cod)

    # uncurried domain, ick
    elif isinstance(ty, (list, tuple)):
        return map(partial(resolve, ctx), ty)

    # type variables
    else:
        # Either a named constant or a polymorphic type variable
        return ctx.get(ty, N.pv(ty))

# Instantiate the function definition in the scope of a type instance
# and the class parameters.
def instantiate(lctx, sig):
    icod = sig.cod
    outsig = ()

    # parameterized type:
    #   C x0 x1 ...
    if len(icod) > 1:
        con = icod[0] # constructor
        args = icod[1]

        outparams = []
        for elt in args:
            outparams += (elt, lctx[elt]),

        outsig = (con, outparams)

    # simple type
    #   x0
    else:
        ident = icod[0]
        outsig = (lctx[ident],)

    if opts.ddump_inst:
        debug_passes.dump_instance(sig, outsig)

    # TODO: we don't want to just scrape the cod
    return outsig

def map_sorts(lctx, gctx, ref):
    if isinstance(ref, N.refsort):
        sym = ref.ref
        if sym in lctx:
            return lctx[sym]
        elif ref in gctx:
            return gctx[sym]
        else:
            raise NameError, "Sort cannot resolve reference '%s'" % sym
    else:
        return ref


#------------------------------------------------------------------------
# Constrution
#------------------------------------------------------------------------

# Take a maping of functions in the body of the trait
#
# trait T[t]:
#   f :: a -> b
#   g :: c -> d
#
# impl T[A]:
#   f = fa
#   g = ga
#
# impl T[B]:
#   f = fb
#   g = fb
#
# f' :: d -> ( a -> b )
#
# d = match x with {
#    A -> fa
#    B -> fb
# }

# Traits resolution is defined by the following

# P ∪ Γ ⊢ x :: t

# where P is the local class mapping.

#      P ∪ Γ ∪ a ⊢ x :: a
# -------------------------------   [ Trait Introduction ]
#     P ∪  Γ ⊢  t :: a => b

# P ∪ Γ ⊢ x :: a => b     P ⊢ a
# -------------------------------   [ Trait Elimination ]
#     P ∪  Γ ⊢  t :: b

# Trait resolution

# Given the trait signature P in the typing context
#
# x P ∪ Γ
# Substitution: x in A[s / t]

def build_toplevel(ast):
    imports = []

    # resolve all imports first
    for stmt in ast:
        if isinstance(stmt, N.imprt):
            imports.append(stmt)

    # -- namespace magic here --

    for stmt in ast:
        if isinstance(stmt, N.mod):
            for decl in stmt.body:
                # class type
                if isinstance(decl, N.tclass):
                    yield decl.name, Class(decl.name, decl.params, decl.body)

                # instance type
                elif isinstance(decl, N.instance):
                    yield decl.name, ClassInst(decl.name, decl.params, decl.body)

                # typest definition
                elif isinstance(decl, N.typeset):
                    yield decl.name, Typeset(decl.types)

                # sort definition
                elif isinstance(decl, N.sortdef):
                    pass

                else:
                    raise AssertionError

def build_matcher(anonymous_ns):
    for fn, impls in anonymous_ns.iteritems():
        yield fn, Matcher(fn, impls)


class Kind(object):
    def __repr__(self):
        return ' * '

#------------------------------------------------------------------------
# Matchers
#------------------------------------------------------------------------

match_line = """
    %s => %s
"""

match_template = """
%s = match {
    %s
}
"""

class Matcher(object):

    def __init__(self, fn, impls):
        self.fn = fn
        self.impls = impls

    def __call__(self, ty):
        con = ty[0]
        args = ty[1]
        ctx = dict(args)

        for ity, impl in self.impls:
            if ity == con:
                return instantiate(ctx, impl)

    def __repr__(self):
        options = '\n'.join(match_line % impl for impl in self.impls)
        return match_template % (self.fn, options)

def build_def(sig):
    return Definition(Signature(*sig))

#------------------------------------------------------------------------
# Class Children
#------------------------------------------------------------------------

#------------------------------------------------------------------------
# Module
#------------------------------------------------------------------------

def build_module(a):
    # First Pass, gather declarations
    # -------------------------------
    decls = list(build_toplevel(a))

    # classes must have unique names
    classes   = dict(
        (name, inst)
        for name, inst in decls
        if isinstance(inst, Class)
    )

    # instances are in terms of pre-defined classes
    instances = list(
        (name, inst)
        for name, inst in decls
        if isinstance(inst, ClassInst)
    )

    # Second Pass, resolve dependencies and resolve signature for
    # instance types, build the specializers for the bound methods and
    # the anonymous specializers
    bound = defaultdict(dict)
    anonymous = defaultdict(list)

    for clsname, clsinst in instances:
        if clsname in classes:
            derived = classes[clsname]

            newcls = deepcopy(clsinst)
            newcls.cls = derived

            ctx = {}
            ctx.update(pysorts)

            # concrete
            if len(derived.params) == 1:
                ctx[derived.params[0]] = newcls.params

            # parameterized
            else:
                assert isinstance(newcls.params, tuple)

                if len(derived.params) != len(newcls.params):
                    raise ValueError, 'Wrong number of instance parameters for %s, %s ' % (clsname, newcls.params)

                # trait Foo[A x0 x1 x2 ... ]:
                #           |  |  |  |
                #           v  v  v  v
                # impl Foo[ T y0 y1 y2 ... ]:
                #              |
                #              v
                #     {A: T, x0: y0, ... }

                for dp, np in zip(derived.params, newcls.params):
                    ctx[dp] = np

            #------------------------------------------------------------------
            # Class Members
            #------------------------------------------------------------------

            for fn in newcls.defs:

                if isinstance(fn, N.fnimp):
                    try:
                        typ = derived.defs[fn.name]
                    except KeyError:
                        raise TypeError, 'No corresponding signature for %s in class %s' % (fn.name, clsname)

                    dom = map(partial(resolve, ctx), typ.sig.dom)
                    cod = resolve(ctx, typ.sig.cod)
                    ast = fn.node # value-level implementation

                    sig = Signature(dom, cod, ast)
                    #print fn.name, '::', sig.show()
                    #print fn.name, '=', fn.node
                    #print '\n'

                    head = newcls.params

                    # TODO: FIXME this is horrible to solve
                    # polymorphic type variable when we just want
                    # single head value, in this case we *must
                    # have* a constraint over the free variable
                    if len(head) > 1:
                        if len(head[1]) == 1:
                            head = head[0]

                    # good
                    if fn.name not in bound[head]:
                        bound[head][fn.name] = sig
                    # bad
                    else:
                        raise TypeError, 'Overlapping instance for %s in class %s' % (fn.name, clsname)

                    anonymous[fn.name].append((head, sig))

                elif isinstance(fn, N.opimp):
                    sym = fn.op.strip('_')
                    magic = infix_map[sym]
                    ast = fn.node

                    try:
                        typ = derived.defs[magic]
                    except KeyError:
                        raise TypeError, 'No corresponding signature for %s in class %s' % (fn.op, clsname)

                    dom = map(partial(resolve, ctx), typ.sig.dom)
                    cod = resolve(ctx, typ.sig.cod)

                    #sresolve = partial(map_sorts, ctx, pysorts)
                    #resolved = map(sresolve, ast)

                    sig = Signature(dom, cod, ast)

                    head = newcls.params[0]
                    bound[head][magic] = sig

                    # operators can only be bound

                elif isinstance(fn, N.opalias):
                    raise ValueError, 'Implementations may not define aliases'

                elif isinstance(fn, N.foreign):
                    # XXX
                    pass

                else:
                    raise NotImplementedError

            #------------------------------------------------------------------
            # Class Aliases
            #------------------------------------------------------------------

            for sym in derived.aliases:
                # only support operator aliasing for now
                alias = derived.aliases[sym]
                try:
                    bound[head][sym] = bound[head][alias]
                except KeyError:
                    raise NameError, 'Operator is alised but with no corresponding definition: %s' % alias

                if opts.ddump_alias:
                    debug_passes.debug_aliases(derived, newcls, sym, alias)

    return Module(classes, instances, bound, anonymous)

#------------------------------------------------------------------------
# Modules
#------------------------------------------------------------------------

class Module(object):
    def __init__(self, cls, impls, bound, anon):

        self.cls = cls
        self.impls = impls

        self.bound_ns = bound
        self.anon_ns = dict(build_matcher(anon))
        self.anon_refs = dict(anon)

    def resolve_adhoc(self, fn, value):
        return self.anon_ns[fn](value)

    def resolve_bound(self, sym, defn=None):
        # give all definitions for the symbol
        if not defn:
            return self.bound_ns[sym].items()
        else:
            return self.bound_ns[sym][defn]

    def show(self):
        D = 4
        # --------------------------

        if opts.ddump_classes:
            debug_passes.debug_classes(self.cls)

        # --------------------------

        if opts.ddump_bound:
            debug_passes.debug_bound(self.cls)

        # --------------------------
        if opts.ddump_adhoc:
            debug_passes.debug_adhoc(self.anon_refs.items())

#------------------------------------------------------------------------
# Type Class Instance
#------------------------------------------------------------------------

class Typeset(object):

    def __init__(self, types):
        self.types = types

    def show(self):
        pass

class ClassInst(object):

    def __init__(self, cls, params, decls, constrs=None):
        self.cls = cls
        self.params = params
        self.defs = decls
        self.sorts = []
        self.constraints = {}

        if decls:
            pass

    def show(self):
        return ('instance', (
            'cls'    , self.cls,
            'sorts'  , self.sorts,
            'params' , self.params,
            'defs'   , self.defs
        ))

#------------------------------------------------------------------------
# Type Classes
#------------------------------------------------------------------------

class Class(object):
    def __init__(self, name, params, decls):
        self.name = name
        self.params = params
        self.defs = {}
        self.sorts = []
        self.aliases = {}

        if decls:
            for decl in decls:
                if isinstance(decl, N.fndef):
                    name, sig = decl
                    self.defs[name] = build_def(sig)

                elif isinstance(decl, N.opdef):
                    op, sig = decl
                    sym = op.strip('_')
                    magic = infix_map[sym]
                    self.defs[magic] = build_def(sig)

                elif isinstance(decl, N.opalias):
                    op, ref = decl
                    sym = op.strip('_')
                    magic = infix_map[sym]
                    try:
                        self.aliases[magic] = ref
                        #self.defs[magic] = self.defs[ref]
                        pass
                    except KeyError:
                        raise NameError, \
                            "Operator alias cannot resolve reference '%s'" % ref

                elif isinstance(decl, N.opimp):
                    # XXX
                    pass

                elif isinstance(decl, N.eldef):
                    # XXX
                    pass

                elif isinstance(decl, N.tydef):
                    # XXX
                    pass

                elif isinstance(decl, N.fnimp):
                    raise ValueError, 'Implementation not allowed in class'

                else:
                    raise NotImplementedError

    def show(self):
        return (
            'name'   , self.name,
            'sorts'  , self.sorts,
            'params' , self.params,
            'defs'   , self.defs.items(),
        )

#------------------------------------------------------------------------
# Type Definitions
#------------------------------------------------------------------------

class Definition(object):
    def __init__(self, sig):
        self.sig = sig

    def show(self):
        return ('sig', self.sig.show())

class Signature(object):

    def __init__(self, dom, cod, node=None):
        self.dom = dom
        self.cod = cod
        self.node = node
        # class constraints
        self.constraints = {}

    def show(self):
        # uncurred domain so join with (->)
        def _show(ty):
            if isinstance(ty, N.fn):
                return repr(ty)
            if isinstance(ty, N.pt):
                return repr(ty)
            elif isinstance(ty, tuple):
                return ' '.join(ty)
            elif isinstance(ty, list):
                return repr(ty)
            elif isinstance(ty, basestring):
                return ty

        dom = '%s' % ', '.join(map(_show, self.dom))
        cod = _show(self.cod)
        return '%s --> %s' % (dom, cod)

    def __repr__(self):
        return self.show()


def format_ty(ty):
    if isparametric(ty):
        return '%s %s' % (ty[0], ' '.join([a[1] for a in ty[1]]))
    elif issimple(ty):
        return '%s' % (ty[0],)
    else:
        raise AssertionError
