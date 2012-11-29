"""
Execution plans.
"""
import string
import numpy as np
from mmap import PAGESIZE
from pprint import pprint

from collections import namedtuple
from ndtable.expr.graph import OP, APP, VAL
from ndtable.idx import Indexable
from ndtable.byteproto import CONTIGUOUS, READ

L2SIZE = 2**17
L3SIZE = 2**20

L2   = 1
L3   = 2
PAGE = 3

map_kernels = {
    ('add', 'i', 'i'): ''
}

reduce_kernels = {
    ('sum', 'i', 'i'): ''
}

special_kernels = {
}

#------------------------------------------------------------------------
# Toy Kernels
#------------------------------------------------------------------------

def add_scalars(a,b,o):
    res = np.empty(1, o)
    np.add(a,b,res)
    return res

def add_incore(dd_a, ds_a, dd_b, ds_b, ds_o):
    res = np.empty(ds_o)
    np.add(ds_a.getbuffer(), ds_b.getbuffer(), res)
    return res

# chunks = [((0,1024), (0, 1024))], a range for each of the
# operands
def add_outcore(dd_a, ds_a, dd_b, ds_b, ds_o, chunks):
    res = np.empty(ds_o) # may be a scalar

    for ca_start, ca_end, cb_start, cb_end in chunks:
        np.add(
            ds_a.getchunk(ca_start, ca_end),
            ds_b.getchunk(cb_start, cb_end),
        res)

    return res

#------------------------------------------------------------------------
# Plans
#------------------------------------------------------------------------

class Instruction(object):
    def __init__(self, op, ret=None, *args):
        '%1 = op ...'

        self.op = op
        self.args = args

        self.ret = ret

    def __repr__(self):
        # with output types
        if self.ret:
            return self.ret + ' = ' + ' '.join((self.op,) + self.args)
        # purely side effectful
        else:
            return ' '.join([self.op,] + map(repr, self.args))

class Constant(object):
    def __init__(self, val):
        self.val = val

class BlockPlan(object):

    def __init__(self, operands, align=L2):
        if align == L2:
            self.align = L2SIZE
        elif align == L3:
            self.align = L3SIZE
        elif align == PAGE:
            self.align = PAGESIZE
        else:
            raise NotImplementedError

        self.operands = operands

    def generate(self):
        if len(self.operands) == 1:
            return self.generate1()
        else:
            return self.generate2()

    def generate2(self):
        return zip(
            list(self.chunk_for(self.operands[0])),
            list(self.chunk_for(self.operands[1]))
        )

    def generate1(self):
        return list(self.chunk_for(self.operands[0]))

    def chunk_for(self, o):
        """ Generate a list of blocksizes for the operand given the total
        number of bytes specified by the byte descriptor.
        """
        last = 0

        # it's a tiny chunk thats less than the alignment, so
        # it's trivial
        if o.nbytes < self.align:
            yield (0, o.nbytes)
            raise StopIteration

        # aligned chunks
        for csize in xrange(0, o.nbytes, self.align):
            last = csize
            yield (last, csize)

        # leftovers
        if last < o.nbytes:
            yield (last, o.nbytes)
        else:
            raise StopIteration

def tmps():
    for i in string.letters:
        yield '%' + i

def _generate(nodes, _locals, _retvals, tmpvars):
    for op in nodes:
        blocked = False
        largs = []

        for arg in op.children:

            if arg.kind == APP:
                largs.append(_retvals[arg.operator])

            elif arg.kind == VAL:
                #-----------------
                # Arrays & Tables
                #-----------------
                if isinstance(arg, Indexable):
                    # Read the data descriptor for the array or
                    # table in question.

                    # If the ByteProvider supports contigious
                    # reading then we're in luck just do the
                    # operation in core
                    if arg.data.has_op(CONTIGUOUS, READ):
                        largs.append(arg.data.read_desc())

                    # Otherwise we have to figure out the
                    # BlockPlan ( term from Travis ) to figure
                    # out how to shuffle bytes chunkwise given
                    # the bounds and chunk alignment of the data
                    # descriptor
                    else:
                        blocked = True
                        largs.append(arg.data.read_desc())

                #-----------
                # Constants
                #-----------
                elif isinstance(arg.val, (int, long, float)):
                    largs.append(str(arg.data.pyobject))

                #-------------------
                # Named Expressions
                #-------------------
                else:
                    _locals[op.val]

        # blocked execution ( out-of-core array expressions )
        if blocked:
            # XXX
            dummy_size = '4096'
            tmpvar = next(tmpvars)
            _retvals[op] = tmpvar

            bplan = BlockPlan(largs, align=PAGE)
            for a,b in bplan.generate():
                yield Instruction('alloca_chunk', tmpvar, dummy_size)
                yield Instruction(op.__class__.__name__, None, *(largs + [tmpvar]))

        # blocked execution ( scalar operations, in-core array expressions)
        else:
            # XXX
            dummy_size = '4096'
            tmpvar = next(tmpvars)
            _retvals[op] = tmpvar
            yield Instruction('alloca', tmpvar, dummy_size)
            yield Instruction(op.__class__.__name__, None, *(largs + [tmpvar]))

def generate(graph, variables, kernels):
    # The variables come in topologically sorted, so we just
    # have to preserve that order
    _locals = {}
    _retvals = {}

    res = list(_generate(graph, _locals, _retvals, tmps()))
    return res

def explain(plan):
    pass

dd_t = namedtuple('datadescriptor', 'nbytes, nchunks, chunksize')
op_t = namedtuple('operand', 'dd')

if __name__ == '__main__':
    op1 = op_t(dd_t(5000, 0, 0))
    op2 = op_t(dd_t(5000, 0, 0))

    print BlockPlan([op1, op2], align=PAGE).generate()
