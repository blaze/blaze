from __future__ import absolute_import

import types
import re
from blaze.datashape.typesets import TypeSet, matches_typeset

from blaze.datashape import (DataShape, from_numba_str, to_numba, broadcastable)
from ...io.datadescriptor.blaze_func_descriptor import BlazeFuncDeprecatedDescriptor
from blaze.py2help import _strtypes, PY2
from .. import llvm_array as lla
from .blaze_kernels import BlazeElementKernel, frompyfunc
from .kernel_tree import Argument, KernelTree

def process_signature(ranksignature):
    """
    Convert list of comma-separated strings into a list of integers showing
    the rank of each argument and a list of sets of tuples.  Each set
    shows dimensions of arguments that must match by indicating a 2-tuple
    where the first integer is the argument number (output is argument -1) and
    the second integer is the dimension

    Examples
    --------

    >>> from blaze.bkernel.blfuncs import process_signature
    >>> process_signature(['M,L', 'L,K', 'K', ''])
    ([2, 2, 1, 0], [set([(2, 0), (1, 1)]), set([(0, 1), (1, 0)])])
    """
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

# Parse numba-style calling string convention
#  and construct dshapes
regex = re.compile('([^(]*)[(]([^)]*)[)]')
def to_dshapes(mystr, output=False):
    ma = regex.match(mystr)
    if ma is None:
        raise ValueError("Cannot understand signature string % s" % mystr)
    else:
        ret, args = ma.groups()

    result = tuple(from_numba_str(x) for x in args.split(','))
    if output:
        result += (from_numba_str(ret),)
    return result

def convert_kernel(value, dshapes=None):
    template = None
    from llvm.core import FunctionType
    if isinstance(value, tuple): # Ex: ('cpp', source) or ('f8(f8,f8)', _add)
        if len(value) == 2 and isinstance(value[1], types.FunctionType) and \
                   isinstance(value[0], _strtypes):
                if '*' in value[0]:
                    return None, (to_dshapes(value[0]), value[1])
                krnl = frompyfunc(value[1], value[0])  # use Numba
        elif len(value) == 2 and isinstance(value[1], _strtypes) and \
                 isinstance(value[0], _strtypes):
            krnl = _convert_string(value[0], value[1]) # Use blaze_kernels.from<val0>
        else:
            raise TypeError("Cannot parse kernel specification %s" % value)
    elif isinstance(value, types.FunctionType):
        # Called when a function is used for value directly
        # dshapes must be present as the mapping and as datashapes
        istemplate = any(isinstance(ds, TypeSet) for ds in dshapes[:-1])
        if istemplate:
            krnl = None
            template = (dshapes[:-1], value)
        else:
            args = ','.join(str(to_numba(ds)) for ds in dshapes[:-1])
            signature = '{0}({1})'.format(str(to_numba(dshapes[-1])), args)
            krnl = frompyfunc(value, signature, dshapes=dshapes)
    elif isinstance(value, FunctionType):
        # Called the LLVM Function is used in directly
        krnl = BlazeElementKernel(value, dshapes=dshapes)
    else:
        raise TypeError("Cannot convert value = %s and dshapes = %s" % (value, dshapes))

    return krnl, template

def process_typetable(typetable):
    """
    Process type-table dictionary which maps a signature list with
      (input-type1, input-type2, output_type) to a kernel into a
      lookup-table dictionary which maps an input-only signature list
      to a kernel matching those inputs.  The output
      is placed with a tuple of the output-type plus the signature
    So far it assumes the types all have the same rank
      and deduces the signature from the first kernel found
    Also allows the typetable to have "templates" which don't resolve to
      kernels and are used if no matching kernel can be found.
      templates are list of 2-tuple (input signature data-shape, template)
      Numba will be used to jit the template at call-time to create a
      BlazeElementKernel.   The input signature is a tuple
      of data-shape objects and TypeSets
    """
    newtable = {}
    templates = []
    if isinstance(typetable, list):
        for item in typetable:
            krnl, template = convert_kernel(item)
            if template is None:
                in_shapes = krnl.dshapes[:-1]
                newtable[in_shapes] = krnl
            else:
                templates.append(template)
    else:
        for key, value in typetable.items():
            if not isinstance(value, BlazeElementKernel):
                value, template = convert_kernel(value, dshapes=key)
            if template is None:
                in_shapes = value.dshapes[:-1]
                newtable[in_shapes] = value
            else:
                templates.append(template)

    # FIXME:
    #   Assumes the same ranklist and connections for all the keys
    if len(newtable.values()) > 0:
        key = next(iter(newtable.values())).dshapes
        ranklist, connections = get_signature(key)
    else: # Currently templates are all rank-0
        ranklist = [0]*len(templates[0][0])
        connections = []

    return ranklist, connections, newtable, templates

# Define the Blaze Function
#   * A Blaze Function is a callable that takes Concrete Arrays and returns
#        Deferred Concrete Arrays
#   * At the core of the Blaze Function is a kernel which is a type-resolved
#        element-wise expression graph where elements can be any contiguous
#        primitive type (right-most part of the data-shape)
#   * Kernels have a type signature which we break up into the rank-signature
#       and the primitive type signature because a BlazeFuncDeprecated will have one
#       rank-signature but possibly multiple primitive type signatures.
#   * Example BlazeFuncDeprecateds are sin, svd, eig, fft, sum, prod, inner1d, add, mul
#       etc --- kernels all work on in-memory "elements"

class BlazeFuncDeprecated(object):
    # DEPRECATION NOTE:
    #   This particular blaze func class is being deprecated in favour of
    #   a new implementation, using the pykit system. Functionality will
    #   be moved/copied from here as needed until this class can be removed.
    def __init__(self, name, typetable=None, template=None, inouts=[]):
        """
        Construct a Blaze Function from a rank-signature and keyword arguments.

        The typetable is a dictionary with keys a tuple of types
        and values as corresponding BlazeScalarKernel objects.  The
        tuple of types has Input types first with the Output Type last

        Arguments
        =========
        name  : string
            Name of the Blaze Function.

        typetable : dict
            Dictionary mapping argument types to blaze kernels.

            The kernels must all be BlazeElementKernel instances or
            convertible to BlazeElementKernel via the following
            mechanisms:
                python-function:  converted via numba
                llvm-function:    directly wrapped
                ctypes-function:  wrapped via an llvm function call

        inouts : list of integers
            A list of the parameter indices which may be written to
            in addition to read from. (NotImplemented)
        """
        self.name = name
        if typetable is None:
            self.ranks = None
            self.rankconnect = []
            self.dispatch = {}
            self.templates = []
        else:
            res = process_typetable(typetable)
            self.ranks, self.rankconnect, self.dispatch, self.templates = res
        self.inouts = inouts
        self._add_template(template)


    def _add_template(self, template):
        if template is None:
            return
        if lla.isiterable(template):
            for temp in template:
                self._add_template_sub(temp)
        else:
            self._add_template_sub(template)

    def _add_template_sub(self, template):
        if isinstance(template, tuple):
            self.add_template(template[0], signature=template[1])
        else:
            self.add_template(template)

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
            if krnl is not None or test_rank==0:
                break
            test_rank -= 1
            mtypes = tuple(ds.subarray(1) for ds in mtypes)

        # Templates Only works for "measures"
        if krnl is None:
            measures = tuple(ds.measure for ds in types)
            for sig, template in self.templates:
                if sig == measures or matches_typeset(measures, sig):
                    krnl = frompyfunc(template, [to_numba(x) for x in measures])
                    self.dispatch[measures] = krnl
                    break

        if krnl is None:
            raise ValueError("Did not find matching kernel for " + str(mtypes))

        return krnl

    def add_funcs(self, value):
        res = process_typetable(value)
        ranklist, connections, newtable, templates = res
        if self.ranks is None:
            self.ranks = ranklist
            self.rankconnect = connections

        self.dispatch.update(newtable)
        self.templates.extend(templates)

    def add_template(self, func, signature=None):
        if signature is None:
            fc = func.func_code if PY2 else func.__code__
            signature = '*(%s)' % (','.join(['*']*fc.co_argcount))

        keysig = to_dshapes(signature)
        self.templates.append((keysig, func))

        # All templates are 0-rank
        if self.ranks is None:
            self.ranks = [0]*len(keysig)

