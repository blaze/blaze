from __future__ import absolute_import

# A Blaze Element Kernel is a wrapper around an LLVM Function with a
#    particular signature.
#    The kinds of argument types are simple, ptr, and array.
#    A kernel kind is a tuple of input kinds followed by the output kind
#
#    simple:  out_type @func(in1_type %a, in2_type %b)
#    ptrs:  void @func(in1_type * %a, in2_type * %b, out_type * %out)
#    array:  void @func(in1_array * %a, in2_array * %b, out_array * %out)
#
# We use a simple array type definition at this level for arrays
# struct {
#    eltype *data;
#    int nd;
#    diminfo dims[nd];
#} array
#
# struct {
#   intp dim;
#   intp stride;
#} diminfo
#

import sys

import llvm.core as lc
from llvm.core import Type, Function

void_type = Type.void()
int_type = Type.int()
intp_type = Type.int(8) if sys.maxsize > 2**32 else Type.int(4)
diminfo_type = Type.struct([
    intp_type,    # shape
    intp_type     # stride
    ], name='diminfo')

array_type = lambda el_type: Type.struct([
    Type.pointer(el_type),       # data
    int_type,                    # nd
    Type.array(diminfo_type, 0)  # dims[nd]  variable-length struct
    ])

SCALAR = 0
POINTER = 1
ARRAY = 2

def isarray(arr):
    if not isinstance(arr, lc.StructType):
        return False
    if arr.element_count != 3 or \
        not isinstance(arr.elements[0], lc.PointerType) or \
        not arr.elements[1] == int_type or \
        not isinstance(arr.elements[2], lc.ArrayType):
        return False
    shapeinfo = arr.elements[2]
    if not shapeinfo.element == diminfo_type:
        return False
    return True

# A wrapper around an LLVM Function object
class BlazeElementKernel(object):
    def __init__(self, func):
        if not isinstance(func, Function):
            raise ValueError("Function should be an LLVM Function."\
                                " Try a converter method.")
        self.func = func
        func_type = func.type.pointee
        kindlist = [None]*func_type.arg_count
        if not (func_type.return_type == void_type):  # Scalar output
            kindlist += [SCALAR]

        ranks = [0]*kindlist
        for i, arg in enumerate(func_type.args):
            if not isinstance(arg, lc.PointerType):
                kindlist[i] = SCALAR
            elif isarray(arg.pointee):
                kindlist[i] = ARRAY
                kindlist[i] = None  # unknown
            else:
                kindlist[i] = POINTER
        self.kind = tuple(kindlist)
        self.ranks = ranks

    @property
    def nin(self):
        return len(self.kind)-1

    # Currently only works for scalar kernels
    @staticmethod
    def frompyfunc(pyfunc, signature):
        import numba
        func = BlazeElementKernel(numba.jit(signature)(pyfunc).lfunc)
        func.name = pyfunc.func_name
        return func

    @staticmethod
    def fromblir(str):
        pass

    @staticmethod
    def fromcffi(cffifunc):
        pass

    @staticmethod
    def fromctypes(ctypefunc):
        pass

    @staticmethod
    def fromcfunc(cfunc):
        pass

<<<<<<< HEAD
    def create_wrapper_kernel(inrank, outrank):
        """Take the current kernel of inrank and create a new kernel of
        outrank that calls call the current kernel multiple times as needed

        Example (let rn == rank-n)
          We need an r2, r2 -> r2 kernel and we have an r0, r0 -> r0
          kernel.
=======
    def create_wrapper_kernel(input_ranks, output_rank):
        """Take the current kernel and available input argument ranks
         and create a new kernel that matches the required output rank 
         by using the current kernel multiple-times if necessary. 

         So that we can create a simple call stack
>>>>>>> Check to ensure large-enough arrays for kernel rank and being work of creating wrappers and fusers.

        Example: (let rn == rank-n) 
          We need an r2, r2 -> r2 kernel and we have an r1, r1 -> r1
          kernel.    

          We create a kernel with rank r2, r2 -> r2 that does the equivalent of

          for i in range(n0):
              out[i] = inner_kernel(in0[i], in1[i])
        """

        raise NotImplementedError


def fuse_kerneltree(tree):
    """Fuse the kernel tree into a single kernel object with the common names
     
    Define a variable for each unique node that is not a scalar. 

    add(multiply(b,c),subtract(d,f))

    var tmp0 = multiply(b,c)
    var tmp1 = subtract(d,f)

    return add(tmp0, tmp1)

    var tmp0;
    var tmp1;

    multiply(b,c,&tmp0)
    subtract(d,f,&tmp1)

    add(tmp0, tmp1, &res)

    """
