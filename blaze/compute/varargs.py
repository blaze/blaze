from abc import ABCMeta

from datashape.py2help import with_metaclass
from toolz import unique, memoize

from .core import compute_up


def flatten(ts):
    """Flatten a (potentially nested) sequence of tuples into just scalars.
    """
    for t in ts:
        if isinstance(t, tuple):
            for flat in flatten(t):
                yield flat
        else:
            yield t


class _VarArgsMeta(type):
    """Metaclass for ``VarArgs`` which allows us to create type-checked
    subclasses using ``__getitem__`` syntax.
    """
    @staticmethod
    @memoize
    def __getitem__(types):
        if not isinstance(types, tuple):
            # only a single type is given, promote to a sequence
            types = types,
        else:
            # flatten the types out and remove any duplicates
            types = tuple(unique(flatten(types)))

        class TypedVarArgs(with_metaclass(_TypedVarArgsMeta, VarArgs)):
            _types = types

        name = 'VarArgs[%s]' % ', '.join(
            getattr(type_, '__name__', str(type_)) for type_ in types
        )
        TypedVarArgs.__name__ = TypedVarArgs.__qualname__ = name
        return TypedVarArgs


class _TypedVarArgsMeta(ABCMeta, _VarArgsMeta):
    """Metaclass for type-checked subclasses of ``VarArgs`` which implement
    ``__subclasscheck__`` for type dispatch.
    """
    def __subclasscheck__(self, other):
        self_types = self._types
        return (
            isinstance(other, type(self)) and
            all(issubclass(type_, self_types) for type_ in other._types)
        )


class VarArgs(with_metaclass(_VarArgsMeta, tuple)):
    """An immutable, typed variadic sequence.

    Parameters
    ----------
    args : iterable[any]
        The values to create a ``VarArgs`` from.

    Notes
    -----
    To create a type-checked subclass, use: ``VarArgs[Type1, Type2, ...]``.
    """
    def __new__(cls, args):
        args = tuple(args)
        if cls is VarArgs:
            # if ``VarArgs`` is created directly infer the type from the args
            return cls[tuple(map(type, args))](args)

        # this is the __new__ of the subclass, typecheck at construction time:
        for n, arg in enumerate(args):
            if not isinstance(arg, cls._types):
                raise TypeError(
                    'invalid type %s at index %d, must be one of %s' % (
                        type(arg),
                        n,
                        cls._types,
                    ),
                )
        return super(VarArgs, cls).__new__(cls, args)

    def __repr__(self):
        return '%s([%s])' % (
            type(self).__name__,
            super(VarArgs, self).__repr__()[1:-1],  # cut off the parens
        )


def _varargs_materialize(expr, *args, **kwargs):
    """compute_up dispatch for VarArgs. We will dynamically register this
    with multipledispatch based on the length of the varargs.
    """
    return VarArgs(args)


def register_varargs_arity(arity, type_=object):
    """Register a compute_up handler for a varargs of length ``arity``

    Parameters
    ----------
    arity : int
        The arity of the varargs dispatcher to register.
    type_ : type, optional
        The type to register the dispatch with. This defaults to ``object``.
    """
    cache = register_varargs_arity._cache.setdefault(type_, set())
    if arity not in cache:
        # we cache these because it can take almost 2 seconds to add a function
        # to the compute_up dispatcher. If we have already registered
        # a handler for the given arity and type then just return since this
        # is a nop.

        # lazy import to prevent cycles
        from blaze.expr.expressions import VarArgsExpr

        # use a cache here so that we don't need to reorder the compute_up
        # dispatcher every time we create a VarArgsExpr.
        compute_up.register(*(VarArgsExpr,) + (type_,) * arity)(
            _varargs_materialize,
        )
        cache.add(arity)
register_varargs_arity._cache = {}


# Pre-register arities [0, 6] while we are halting multiple dispatch ordering as
# a performance improvement.
# NOTE: this number was picked because it seemed like a good balance of slowing
# down later dispatches with speeding up common cases. We could tune this
# better in the future.
for n in range(7):
    register_varargs_arity(n)
del n
