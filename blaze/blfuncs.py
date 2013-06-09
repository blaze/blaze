from __future__ import absolute_import

import sys
import types

from .datashape.util import broadcastable, to_numba
from .datashape.coretypes import DataShape
from .datadescriptor.blaze_func_descriptor import BlazeFuncDescriptor
from .array import Array
from .cgen.utils import letters
from .blaze_kernels import (Argument, fuse_kerneltree, BlazeElementKernel,
                frompyfunc)
from . import blaze_kernels
from .py3help import dict_iteritems, _strtypes

# A KernelTree is just the bare element-wise kernel functions
# (no arguments).  Any arguments are identified as unique-names
# in an abstract name-space
# All nodes in the kernel tree can also be named
#   or else a unique-name
# from the abstract name-space will be created.
# Each KernelTree has a single llvm module name-space
class KernelTree(object):
    _stream_of_unique_names = letters()
    _stream_of_unique_kernels = letters()

    _fused = None
    _funcptr = None
    _ctypes = None
    _single_ckernel = None
    _mark = False
    _shape = None
    def __init__(self, kernel, children=[], name=None):
        assert isinstance(kernel, BlazeElementKernel)
        for el in children:
            assert isinstance(el, (KernelTree, Argument))
        self.kernel = kernel
        self.children = children
        if name is None:
            name = 'node_' + next(self._stream_of_unique_names)
        self.name = name

    def _reset_marks(self):
        self._mark = False
        if not self.leafnode:
            for child in self.children:
                if isinstance(child, KernelTree):
                    child._reset_marks()

    def sorted_nodes(self):
        """Return depth-first list of unique KernelTree Nodes.

        The root of the tree will be the last node.
        """
        nodes = []
        self._reset_marks()
        self.visit(nodes)
        return nodes

    @property
    def shape(self):
        if self._shape is None:
            shapeargs = [child.shape for child in self.children]
            self._shape = self.kernel.shapefunc(*shapeargs)
        return self._shape

    @property
    def leafnode(self):
        return all(isinstance(child, Argument) for child in self.children)

    def visit(self, nodes):
        if not self.leafnode:
            for child in self.children:
                if isinstance(child, KernelTree):
                    child.visit(nodes)
        if not self._mark:
            nodes.append(self)
            self._mark = True

    def fuse(self, name=None):
        if self.leafnode:
            self._update_kernelptrs(self)
            return self
        if name is None:
            name = 'flat_' + self.name
        krnlobj, children = fuse_kerneltree(self, name)
        new = KernelTree(krnlobj, children)
        self._update_kernelptrs(new)
        return new

    def _update_kernelptrs(self, elkernel):
        self._fused = elkernel
        kernel = elkernel.kernel
        self._funcptr = self._funcptr or kernel.func_ptr
        self._ctypes = self._ctypes or kernel.ctypes_func
        if all(kind == blaze_kernels.SCALAR for kind in kernel.kinds):
            self._single_ckernel = self._single_ckernel or kernel.single_ckernel

    @property
    def func_ptr(self):
        if self._funcptr is None:
            self.fuse()
        return self._funcptr

    @property
    def ctypes_func(self):
        if self._ctypes is None:
            self.fuse()
        return self._ctypes

    @property
    def single_ckernel(self):
        if self._single_ckernel is None:
            self.fuse()
        return self._single_ckernel

    def adapt(self, newrank, newkind):
        """
        Take this kernel tree and create a new kerneltree adapted
        so that the it can be the input to another element kernel
        with rank newrank and kind newkind
        """
        krnlobj, children = fuse_kerneltree(self, "fused_temp_" + self.name)
        typechar = blaze_kernels.orderchar[newkind[0]]
        new = krnlobj.lift(newrank, typechar)
        children = [child.lift(newrank, newkind[0]) for child in children]
        return KernelTree(new, children)

    def __call__(self, *args):
        return self.ctypes_func(*args)

# Convert list of comma-separated strings into a list of integers showing
#  the rank of each argument and a list of sets of tuples.  Each set
#  shows dimensions of arguments that must match by indicating a 2-tuple
#  where the first integer is the argument number (output is argument -1) and
#  the second integer is the dimension
# Example:  If the rank-signature is ['M,L', 'L,K', 'K', '']
#           then the list of integer ranks is [2, 2, 1, 0]
#           while the list of connections is [{(0,1),(1,0)},{(1,1),(2,0)}]
def process_signature(ranksignature):
    ranklist = [0 if not arg else len(arg.split(',')) for arg in ranksignature]

    varmap = {}
    for i, arg in enumerate(ranksignature):
        if not arg:
            continue
        for k, var in enumerate(arg.split(',')):
            varmap.setdefault(var, []).append((i, k))

    connections = [set(val) for val in varmap.values() if len(val) > 1]
    return ranklist, connections

def get_signature(typetuple):
    sig = [','.join((str(x) for x in arg.shape)) for arg in typetuple]
    return process_signature(sig)

def _convert_string(kind, source):
    try:
        func = getattr(blaze_kernels, 'from%s' % kind)
    except AttributeError:
        return ValueError("No conversion function for %s found." % kind)
    return func(source)

def convert_kernel(value, key=None):
    from llvm.core import FunctionType
    if isinstance(value, tuple):
        if len(value) == 2 and isinstance(value[1], types.FunctionType) and \
                 isinstance(value[0], _strtypes):
                krnl = frompyfunc(value[1], value[0])
        elif len(value) == 2 and isinstance(value[1], _strtypes) and \
                 isinstance(value[0], _strtypes):
            krnl = _convert_string(value[0], value[1])
        else:
            raise TypeError("Cannot parse kernel specification %s" % value)
    elif isinstance(value, types.FunctionType) and key is not None:
        # Requires key to be provided as the mapping.
        args = ','.join(str(to_numba(ds)) for ds in key[:-1])
        signature = '{0}({1})'.format(str(to_numba(key[-1])), args)
        krnl = frompyfunc(value, signature)
        krnl.dshapes = key
    elif isinstance(value, FunctionType):
        krnl = BlazeElementKernel(value)
        if key is not None:
            krnl.dshapes = key
    else:
        raise TypeError("Cannot convert value = %s and key = %s" % (value, key))

    return krnl

# Process type-table dictionary which maps a signature list with
#   (input-type1, input-type2, output_type) to a kernel into a
#   lookup-table dictionary which maps an input-only signature list
#   to a kernel matching those inputs.  The output
#   is placed with a tuple of the output-type plus the signature
# So far it assumes the types all have the same rank
#   and deduces the signature from the first kernel found
def process_typetable(typetable):
    newtable = {}
    if isinstance(typetable, list):
        for item in typetable:
            krnl = convert_kernel(item)
            in_shapes = krnl.dshapes[:-1]
            newtable[in_shapes] = krnl
        key = krnl.dshapes
    else:
        for key, value in typetable.items():
            if not isinstance(value, BlazeElementKernel):
                value = convert_kernel(value, key)
            value.dshapes = key
            in_shapes = value.dshapes[:-1]
            newtable[in_shapes] = value

    # FIXME:
    #   Assumes the same ranklist and connections for all the keys
    ranklist, connections = get_signature(key)

    return ranklist, connections, newtable

# Define the Blaze Function
#   * A Blaze Function is a callable that takes Concrete Arrays and returns
#        Deferred Concrete Arrays
#   * At the core of the Blaze Function is a kernel which is a type-resolved
#        element-wise expression graph where elements can be any contiguous
#        primitive type (right-most part of the data-shape)
#   * Kernels have a type signature which we break up into the rank-signature
#       and the primitive type signature because a BlazeFunc will have one
#       rank-signature but possibly multiple primitive type signatures.
#   * Kernels for a particular type might be inline jitted or loaded from
#       a shared-library --- uses BLIR kernel engine on-top of LLVM.
#   * Example BlazeFuncs are sin, svd, eig, fft, sum, prod, inner1d, add, mul
#       etc --- kernels all work on in-memory "elements"

class BlazeFunc(object):
    def __init__(self, name, typetable, inouts=[]):
        """
        Construct a Blaze Function from a rank-signature and keyword arguments.

        The typetable is a dictionary with keys a tuple of types
        and values as corresponding BlazeScalarKernel objects.  The
        tuple of types has Input types first with the Output Type last

        Arguments
        =========
        name  :  String name of the Blaze Function

        typetable :  dictionary mapping argument types to an implementation
                     kernel which is an instance of a BlazeElementKernel object

                     The kernel may also be several other objects which will
                     be converted to the BlazeElementKernel:
                        python-function:  converted via numba
                        string:           converted via blir
                        llvm-function:    directly wrapped
                        ctypes-function:  wrapped via an llvm function call

                    This may also be a list of tuples which will be interpreted as
                    a dict with the first-argument first converted to dshape depending

        inouts : list of integers corresponding to arguments which are
                  input and output arguments (NotImplemented)
        """
        self.name = name
        res = process_typetable(typetable)
        self.ranks, self.rankconnect, self.dispatch = res
        self.inouts = inouts

    @property
    def nin(self):
        return len(self.ranks)-1

    def compatible(self, args):
        # check for broadcastability
        # TODO: figure out correct types as well
        dshapes = [arg._data.dshape for arg in args]
        return broadcastable(dshapes, self.ranks,
                                rankconnect=self.rankconnect)

    # FIXME: This just does a dumb look-up
    #        assumes input kernels all have the same rank
    def find_best_kernel(self, types):
        mtypes = [ds.sigform() for ds in types]
        ranks = [len(ds)-1 for ds in types]
        test_rank = min(ranks)
        mtypes = tuple(ds.subarray(rank-test_rank)
                      for ds, rank in zip(mtypes, ranks))
        krnl = None
        while test_rank >= 0:
            krnl = self.dispatch.get(mtypes, None)
            if krnl is not None:
                break
            test_rank -= 1
            mtypes = tuple(ds.subarray(1) for ds in mtypes)
        if krnl is None:
            raise ValueError("Did not find matching kernel for " + str(mtypes))
        return krnl

    def __call__(self, *args, **kwds):
        # convert inputs to Arrays
        # build an AST and return Arrays with a Deferred Data Descriptor
        # The eval method of the Array does the actual computation
        #args = map(blaze.asarray, args)


        # FIXME:  The compatible function should unify a suitable
        #          function even if the types are not all the same

        # Find the kernel from the dispatch table
        types = tuple(arr.dshape for arr in args)
        kernel = self.find_best_kernel(types)

        # Check rank-signature compatibility and broadcastability of arguments
        outshape = self.compatible(args)

        # Construct output dshape
        out_type = kernel.dshapes[-1]
        if len(out_type) > 1:
            out_type = out_type.measure

        if len(outshape)==0:
            outdshape = out_type
        else:
            outdshape = DataShape(outshape+(out_type,))

        # Create a new BlazeFuncDescriptor with this
        # kerneltree and a new set of args depending on
        #  unique new arguments in the expression.
        children = []
        for i, arg in enumerate(args):
            data = arg._data
            if isinstance(data, BlazeFuncDescriptor):
                tree = data.kerneltree
                treerank = tree.kernel.ranks[-1]
                argrank = self.ranks[i]
                if argrank != treerank:
                    if argrank > treerank:
                        tree = data.kerneltree.adapt(argrank, kernel.kinds[i])
                    else:
                        raise ValueError("Cannot use rank-%d output "
                            "when rank-%d input is required" % (treerank, argrank))
                children.append(tree)
            else:
                tree_arg = Argument(arg, kernel.kinds[i],
                                    self.ranks[i], kernel.argtypes[i])
                children.append(tree_arg)

        kerneltree = KernelTree(kernel, children)
        data = BlazeFuncDescriptor(kerneltree, outdshape)

        # Construct an Array object from new data descriptor
        # Standard propagation of user-defined meta-data.
        user = {self.name: [arg.user for arg in args]}

        # FIXME:  Check for axes alignment and labels alignment
        axes = args[0].axes
        labels = args[0].labels

        return Array(data, axes, labels, user)




