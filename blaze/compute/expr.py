
"""
Blaze expression graph for deferred evaluation. Each expression node has
an opcode and operands. An operand is a Constant or another expression node.
Each expression node carries a DataShape as type.
"""

from __future__ import absolute_import, division, print_function

from collections import Iterable
from functools import partial
from io import BytesIO
import warnings
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile


from datashape import coretypes as T, dshape

import blaze

# Optional imports for graph visualization.
try:
    import networkx
    have_networkx = True
except ImportError:
    have_networkx = False


#------------------------------------------------------------------------
# Opcodes
#------------------------------------------------------------------------
array = 'array'    # array input
kernel = 'kernel'  # kernel application, carrying the blaze kernel as a
                   # first argument (Constant)

#------------------------------------------------------------------------
# Graph
#------------------------------------------------------------------------
class Config(object):
    max_argument_recursion = 25
    max_argument_len       = 1000
    argument_sample        = 100

conf = Config()


class Op(object):
    """
    Single node in blaze expression graph.

    Attributes
    ----------
    opcode: string
        Kind of the operation, i.e. 'array' or 'kernel'

    uses: [Op]
        Consumers (or parents) of this result. This is useful to keep
        track of, since we always start evaluation from the 'root', and we
        may miss tracking outside uses. However, for correct behaviour, these
        need to be retained
    """

    def __init__(self, opcode, dshape, *args, **metadata):
        self.opcode = opcode
        self.dshape = dshape
        self.uses = []
        self.args = list(args)

        if opcode == 'kernel':
            assert 'kernel' in metadata
            assert 'overload' in metadata
        self.metadata = metadata

        for arg in self.args:
            arg.add_use(self)

    def add_use(self, use):
        self.uses.append(use)

    def __repr__(self):
        opcode = self.opcode
        if opcode == kernel:
            opcode = self.metadata["kernel"]
        metadata = ", ".join(self.metadata)
        return "%s(...){dshape(%s), %s}" % (opcode, self.dshape, metadata)

    def tostring(self):
        subtrees = " -+- ".join(map(str, self.args))
        node = str(self)
        length = max(len(subtrees), len(node))
        return "%s\n%s" % (node.center(len(subtrees) / 2), subtrees.center(length))

# ______________________________________________________________________
# Graph constructors

ArrayOp    = partial(Op, array)

# Kernel application. Associated metadata:
#   kernel: the blaze.function.Kernel that was applied
#   overload: the blaze.overload.Overload that selected for the input args

KernelOp   = partial(Op, kernel)


def construct(bfunc, ctx, overload, args):
    """
    Blaze expression graph construction for deferred evaluation.

    Parameters
    ----------
    bfunc : Blaze Function
        (Overloaded) blaze function representing the operation

    ctx: ExprContext
        Context of the expression

    overload: blaze.overload.Overload
        Instance representing the overloaded function

    args: list
        bfunc parameters
    """
    from .function import BlazeFunc
    assert isinstance(bfunc, BlazeFunc), bfunc

    params = [] # [(graph_term, ExprContext)]

    # -------------------------------------------------
    # Build type unification parameters

    for i, arg in enumerate(args):
        if isinstance(arg, blaze.Array) and arg.expr:
            # Compose new expression using previously constructed expression
            term, context = arg.expr
            if not arg.deferred:
                ctx.add_input(term, arg)
        elif isinstance(arg, blaze.Array):
            term = ArrayOp(arg.dshape)
            ctx.add_input(term, arg)
            empty = ExprContext()
            arg.expr = (term, empty)
        elif not isinstance(arg, blaze.Array):
            term = ArrayOp(T.typeof(arg))

        ctx.terms[term] = arg
        params.append(term)

    assert isinstance(overload.resolved_sig, T.Function)
    restype = dshape(overload.resolved_sig.restype)

    return KernelOp(restype, *params, kernel=bfunc, overload=overload)


def from_value(value):
    return ArrayOp(T.typeof(value), value)


def is_homogeneous(it):
    # Checks Python types for arguments, not to be confused with the
    # datashape types and the operator types!

    head = it[0]
    head_type = type(head)
    return head, all(type(a) == head_type for a in it)


def injest_iterable(args, depth=0, force_homog=False, conf=conf):
    # TODO: Should be 1 stack frame per each recursion so we
    # don't blow up Python trying to parse big structures
    assert isinstance(args, (list, tuple))

    if depth > conf.max_argument_recursion:
        raise RuntimeError(
        "Maximum recursion depth reached while parsing arguments")

    # tuple, list, dictionary, any recursive combination of them
    if isinstance(args, Iterable):
        if len(args) == 0:
            return []

        if len(args) < conf.max_argument_len:
            sample = args[0:conf.argument_sample]

            # If the first 100 elements are type homogenous then
            # it's likely the rest of the iterable is.
            head, is_homog = is_homogeneous(sample)

            if force_homog and not is_homog:
                raise TypeError("Input is not homogenous.")

            # Homogenous Arguments
            # ====================

            if is_homog:
                return [from_value(a) for a in args]

            # Heterogenous Arguments
            # ======================

            # TODO: This will be really really slow, certainly
            # not something we'd want to put in a loop.
            # Optimize later!

            elif not is_homog:
                ret = []
                for a in args:
                    if isinstance(a, (list, tuple)):
                        sub = injest_iterable(a, depth+1)
                        ret.append(sub)
                    else:
                        ret.append(from_value(a))

                return ret

        else:
            raise RuntimeError("""
            Too many dynamic arguments to build expression
            graph. Consider alternative construction.""")


def dump(node, ipython=True):
    """
    Dump the expression graph to either a file or IPython.
    """
    if not networkx:
        warnings.warn("networkx not installed, unable to view graph")
        return

    graph = build_graph(networkx.DiGraph(), node, set())
    if ipython:
        return browser(graph)
    else:
        return view(graph)


def build_graph(graph, term, seen):
    if term in seen:
        return

    seen.add(term)
    for arg in term.args:
        graph.add_edge(term, arg)
        build_graph(graph, arg, seen)

    return graph


def browser(graph):
    from IPython.core.display import Image
    import networkx

    with NamedTemporaryFile(delete=True) as tempdot:
        networkx.write_dot(graph, tempdot.name)
        tempdot.flush()
        p = Popen(['dot', '-Tpng', tempdot.name], stdout=PIPE)
        pngdata = BytesIO(p.communicate()[0]).read()

    return Image(data=pngdata)


def view(self):
    import matplotlib.pyplot as plt
    networkx.draw(self)
    plt.show()


class ExprContext(object):
    """
    Context for blaze graph expressions.

    This keeps track of a mapping between graph expression nodes and the
    concrete data inputs (i.e. blaze Arrays).

    Attributes:
    ===========

    terms: { ArrayOp: Array }
        Mapping from ArrayOp nodes to inputs
    """

    def __init__(self, contexts=[]):
        # Coercion constraints between types with free variables
        self.constraints = []
        self.terms = {} # All terms in the graph, { Array : Op }
        self.params = []

        for ctx in contexts:
            self.constraints.extend(ctx.constraints)
            self.terms.update(ctx.terms)
            self.params.extend(ctx.params)

    def add_input(self, term, data):
        if term not in self.terms:
            self.params.append(term)
        self.terms[term] = data
