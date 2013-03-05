import operator
from functools import partial
from collections import namedtuple, Iterable

from parser import mopen
from typing import instantiate, format_ty

#------------------------------------------------------------------------
# Protocol Checking
#------------------------------------------------------------------------

# * Object Protocol
# * Number Protocol
# * Sequence Protocol
# * Mapping Protocol
# * Iterator Protocol
# * Buffer Protocol
# * Array Protocol

def has_number(x):
    return operator.isNumberType(x)

def has_sequence(x):
    return operator.isSequenceType(x)

def has_mapping(x):
    return operator.isMappingType(x)

def has_iter(x):
    return isinstance(x, Iterable)

def has_callable(x):
    return callable(x)

def has_buffer(x):
    try:
        memoryview(x)
    except TypeError:
        return False
    return True

def has_array(x):
    return hasattr(x, '__array_interface__')

#------------------------------------------------------------------------
# Implicit Operators
#------------------------------------------------------------------------

# Python has implicit implementations for comparison operators such that
# for all objects they are well defined unless you explictly make them
# undefined ... yeah good stuff.

def _repr(o):
    if isinstance(o, Proxy):
        return format_ty(o.typeof)
    else:
        return repr(o)

class ProxyTypeError(TypeError):

    def __init__(self, op, arity, a, b=None):
        self.arity = arity
        self.op = op
        self.a = a
        self.b = b

    def __str__(self):
        if self.arity == 1:
            return "bad operand type(s) for unary %s: '%s'" % (
                (self.op, _repr(self.a)))
        elif self.arity == 2:
            return "unsupported operand type(s) for %s: '%s' and '%s'" % (
                (self.op, _repr(self.a), _repr(self.b)))
        else:
            raise AssertionError

def fail_implicit(op):

    def fail(*ops):
        raise ProxyTypeError(op, len(ops), *ops)

    return fail

implicit_binops = {
    '__gt__' : fail_implicit('>'),
    '__lt__' : fail_implicit('<'),
    '__ge__' : fail_implicit('>='),
    '__le__' : fail_implicit('<='),
    '__eq__' : fail_implicit('=='),
    '__ne__' : fail_implicit('!='),
}

implicit_unops = {
    '__nonzero__' : partial(fail_implicit, 'not'),
}

#------------------------------------------------------------------------
# Prelude
#------------------------------------------------------------------------

core = mopen('blaze.mod')

pysorts = {
    'int'     : 'int',
    'float'   : 'float',
    'complex' : 'complex',
    'index'   : 'tuple',
    'bool'    : 'bool',
    'any'     : 'object',
}

def build_bound_meth(ctx, name, sig):
    def method(self, *args, **kw):
        # since we're bound splice the domain at the
        # first domain element
        adom, bdom = sig.dom[0], sig.dom[1:]
        saturated = len(bdom) == len(args)

        # resolve the codomain of the signature it terms
        # of the proxy _ty variables
        if saturated:
            outsig = instantiate(ctx, sig)
            return Proxy(outsig)
        else:
            # We can achieve a measure of currying with this...
            # which would be neat.
            raise TypeError, '%s, expected %i arguments, got %i' % \
                ( name, len(sig.dom), len(args) + 1 )
    return method

#------------------------------------------------------------------------
# Proxy Utils
#------------------------------------------------------------------------

class ProxyAttributeError(AttributeError):
    pass

def _getattr(self, name):
    return object.__getattribute__(self, name)

def _setattr(self, name, value):
    return object.__setattr__(self, name, value)

#------------------------------------------------------------------------
# Proxy Utils
#------------------------------------------------------------------------

class Proxy(object):
    __slots__ = ["_ty", "_ns", "_ctx", "_node"]

    def __new__(cls, ty, *args, **kwargs):

        assert isinstance(ty, tuple)

        # parameterized type
        if len(ty) > 1:
            con = ty[0]
            ns = core.resolve_bound(con)
            ctx = dict(ty[1])

        # concrete type
        else:
            ns = core.resolve_bound(ty)
            ctx = {}

        # create the proxy object
        proxy = cls._create_proxy(ty, ctx, ns)
        ins = object.__new__(proxy)

        proxy.__init__(ins, ty, ctx)
        _setattr(ins, "_ty", ty)
        _setattr(ins, "_ctx", ctx)

        ctx.update(pysorts)

        return ins

    @classmethod
    def _create_proxy(cls, ty, ctx, ns):
        namespace = {}

        # Capabilities
        # -------------------

        for name, sig in ns:
            meth = build_bound_meth(ctx, name, sig)
            namespace[name] = meth

        namespace['capabilities'] = namespace.keys()

        # Failure for defaults
        # --------------------

        for name in implicit_binops:
            if name not in namespace:
                namespace[name] = implicit_binops[name]

        for name in implicit_unops:
            if name not in namespace:
                namespace[name] = implicit_unops[name]

        return type('Proxy', (cls,), namespace)

    def describe(self):
        """
        Human readable representation of the proxy value
        """
        pass

    @property
    def typeof(self):
        return self._ty

    def __dir__(self):
        """
        Overload the dir() method to list the capabilities of the
        object instead of dumping the __class__.
        """
        return self.capabilities

    def __repr__(self):
        ty = _getattr(self, "_ty")
        if len(ty) > 1:
            return '<Proxy %s>' % format_ty(self._ty)
        else:
            return '<Proxy %s>' % format_ty(self._ty)
