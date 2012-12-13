"""
Unification is a generalization of Numpy broadcasting.

In Numpy we two arrays and broadcast them to yield similar
shaped arrays.

In Blaze we take two arrays with more complex datashapes and
unify the types prescribed by more complicated pattern matching
on the types.

"""

from numpy import promote_types
from blaze.datashape.coretypes import TypeVar
from blaze.expr.typeinference import infer

class Incommensurable(TypeError):
    pass

def unify(sig, concrete=True):
    """
    Unification of Datashapes.
    """
    resolved = infer(sig)
    if all(not isinstance(a, TypeVar) for a in resolved):
        return resolved
