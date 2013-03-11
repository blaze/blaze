# -*- coding: utf-8 -*-

import ast
from _ast import AST

from blirgen import *
from astprint import to_source
from utils import anon

#------------------------------------------------------------------------
# Kernel Manipulation
#------------------------------------------------------------------------

IN    = 0
OUT   = 1
INOUT = 2

class Kernel(object):

    def __init__(self):
        raise NotImplementedError

    @property
    def dimensions(self):
        raise NotImplementedError

    @property
    def retty(self):
        raise NotImplementedError

    @property
    def retval(self):
        raise NotImplementedError

    @property
    def ins(self):
        return [a for i,a in self.arguments if i == IN or i == INOUT]

    @property
    def outs(self):
        return [a for i,a in self.arguments if i == OUT or i == INOUT]

    @property
    def argtys(self):
        return [arg.ty for arg in self.ins]

    def compile(self, **opts):
        """ Compile the kernel into native code. """
        from blaze.blir import compile
        return compile(str(self), **opts)

    def verify(self):
        """ Verify the kernel is well-formed before compilation. """
        shape = None

        # uniform dimensions
        for varg in self.ins:
            if isinstance(varg, VectorArg):
                if not shape:
                    shape = varg.shape
                assert varg.shape == shape
        return True

    def __add__(self, other):
        """ Kernel fusion """
        if isinstance(other, Kernel):
            return fuse(self, other)
        else:
            raise NotImplementedError


class Logic(AST):
    """
    The inner loop logic of the kernel, can be transparently treated as
    if it were a Python AST and manipulated with term-rewrite rules.
    """

    def __init__(self, sexp):
        self.ast = ast.parse(sexp)
        self._fields = self.ast._fields
        self._attributes = self.ast._attributes
        self.body = self.ast.body

    def __str__(self):
        return to_source(self.ast) + ';'

#------------------------------------------------------------------------
# Kernel Parameters
#------------------------------------------------------------------------

class ScalarArg(object):
    def __init__(self, ty, name=None):
        self.ty = ty
        self.name = name


class VectorArg(object):
    def __init__(self, shape, ty, name=None):
        self.ty = ty
        self.name = name
        self.shape = shape

#------------------------------------------------------------------------
# Kernels
#------------------------------------------------------------------------

# Intentionally designed to mirror the PyOpenCL and PyCuda API. These
# high level descriptions of the kernels will allow us to fuse and
# compose kernels symbolically at a high level. LLVM then takes care of
# the instruction level optimizations.

class ElementwiseKernel(Kernel):
    def __init__(self, arguments, operation, name=None):
        self.arguments = arguments
        self.operation = operation
        self.name = name

    @property
    def retty(self):
        # Elemntwise always returns void, if a new array is
        # needed then memory is allocated outside of the kernel
        return 'void'

    @property
    def retval(self):
        return None

    @property
    def dimensions(self):
        # assuming this is verified...
        for varg in self.ins:
            if isinstance(varg, VectorArg):
                for dim in varg.shape:
                    yield (0, dim)
                break

    def __str__(self):
        if hasattr(self, '__cached'):
            return self.__cached

        ivars = {}
        icount = 0

        params = []
        ins = {}
        outs = {}

        _operation = self.operation

        # Preamble
        # --------
        for i, arg in enumerate(self.ins):
            name = arg.name or anon('in')
            param = Arg(arg.ty, name)
            params.append(param)
            ins[i] = param

            _operation = _operation.replace('_in%i' % i, name)

        for i, arg in enumerate(self.outs):
            name = arg.name or anon('out')
            param = Arg(arg.ty, name)
            params.append(param)
            outs[i] = param

            _operation = _operation.replace('_out%s' % i, name)

        inner = Logic(_operation)

        # Loops
        # -----
        for lower, upper in self.dimensions:
            ivar = 'i%s' % icount
            ivars[ivar] = VarDecl('int', ivar, 0)

            inner = For(ivar, Range(lower, upper), Block([inner]))
            icount += 1

        # Return
        # ------
        # always void for elemwise
        ret = []

        # Kernel Body
        decls = ivars.values()
        body = Block(decls + [inner] + ret)

        fn = FuncDef(
            name = self.name or anon('kernel'),
            args = params,
            ret  = self.retty,
            body = body,
        )

        self.__cached = str(fn)
        return self.__cached


class ZipKernel(Kernel):
    def __init__(self, arguments, operationk, name=None):
        self.arguments = arguments
        self.operation = operation

    def __str__(self):
        # TODO
        raise NotImplementedError


class ReductionKernel(Kernel):
    def __init__(self, retty, neutral, reduce_expr, map_expr, arguments, name=None):
        self.retty = retty
        self.neutral = neutral
        self.reduce_expr = reduce_epxr
        self.map_expr = map_expr

    def __str__(self):
        # TODO
        raise NotImplementedError


class ScanKernel(Kernel):
    def __init__(self, retty, arguments, scan_expr, neutral, output_statement, name=None):
        self.retty = retty
        self.neutral = neutral
        self.scan_expr = scan_expr
        self.output_statement = output_statement

    def __str__(self):
        # TODO
        raise NotImplementedError


class OuterKernel(Kernel):
    def __init__(self, retty, arguments, scan_expr, neutral, output_statement, name=None):
        self.retty = retty
        self.neutral = neutral
        self.scan_expr = scan_expr
        self.output_statement = output_statement

    def __str__(self):
        # TODO
        raise NotImplementedError

#------------------------------------------------------------------------
# Kernel Fusion
#------------------------------------------------------------------------

# Naive kernel fusion

def fuse(k1, k2):
    kty1 = k1.__class__
    kty2 = k2.__class__

    if kty1 == ElementwiseKernel and kty2 == ElementwiseKernel:
        return ElementwiseKernel(k1.arguments + k2.arguments, k1.operation)
    else:
        raise NotImplementedError

def compose(k1, k2):
    raise NotImplementedError
