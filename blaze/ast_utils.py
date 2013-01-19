import ast
import pprint
from types import FunctionType, LambdaType

# backproted from Numba

def ast2tree (node, include_attrs = True):
    def _transform(node):
        if isinstance(node, ast.AST):
            fields = ((a, _transform(b))
                      for a, b in ast.iter_fields(node))
            if include_attrs:
                attrs = ((a, _transform(getattr(node, a)))
                         for a in node._attributes
                         if hasattr(node, a))
                return (node.__class__.__name__, dict(fields), dict(attrs))
            return (node.__class__.__name__, dict(fields))
        elif isinstance(node, list):
            return [_transform(x) for x in node]
        return node
    if not isinstance(node, ast.AST):
        raise TypeError('expected AST, got %r' % node.__class__.__name__)
    return _transform(node)

def pformat_ast (node, include_attrs = True, **kws):
    return pprint.pformat(ast2tree(node, include_attrs), **kws)

def dump(node):
    print pformat_ast(node)

def decompile(fn):
    from meta.decompiler import decompile_func
    assert isinstance(fn, (FunctionType, LambdaType)), \
        'Can only decompilefunction type'
    return pformat_ast(decompile_func(fn))
