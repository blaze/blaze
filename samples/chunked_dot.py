""" Sample showing chunked execution of expresions

This sample constructs an expresion to be executed in chunks.
Different aproaches are tested, and it is compared with the
equivalent expresion written in numpy.
"""

import blaze
import blaze.blir as blir
import numpy as np
from time import time
import blaze


# ----------------------------------------------------------------


def _to_blir_type_string(anObject):
    if (isinstance(anObject, blaze.Array)):
        p = anObject.datashape.parameters
        assert(len(p) == 2)
        return 'array[%s]' % 'float'#p[-1].name
    else:
        return 'float' #hardcode ftw


def _gen_blir_decl(name, obj):
    return name + ': ' + _to_blir_type_string(obj) 


def _gen_blir_signature(terms):
    return ',\n\t'.join([_gen_blir_decl(pair[1], pair[0]) 
                      for pair in terms.iteritems()])


# ----------------------------------------------------------------

# Support code to build expresions and convert them to a blir
# function to be executed on each chunk

class Operation(object):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    # operators - used to build an AST of the expresion
    def __add__(self, rhs):
        return Operation('+', self, rhs)

    def __sub__(self, rhs):
        return Operation('-', self, rhs)

    def __mul__(self, rhs):
        return Operation('*', self, rhs)

    def dot(self, rhs):
        return Operation('dot', self, rhs)

    # ------------------------------------------------------------
    # repr
    def __repr__(self):
        return ('Operation(' + repr(self.op) + ', ' 
                + repr(self.lhs) + ', ' 
                + repr(self.rhs) + ')')

    # ------------------------------------------------------------
    # support functions to generate blir code
    def make_terms(self, terms):
        self.lhs.make_terms(terms)
        self.rhs.make_terms(terms)
        return terms

    def gen_blir_expr(self, terms):
        a = self.lhs.gen_blir_expr(terms)
        b = self.rhs.gen_blir_expr(terms)
        return '(' + a + self.op + b + ')'

    def gen_blir(self):
        assert(self.op == 'dot')
        term_array = self.make_terms(set())
        terms = { obj: 'in%d' % i for i, obj in 
                  enumerate(term_array)} 
        code = """
def main(%s, n: int) -> float {
    var float accum = 0.0;
    var int i = 0;
    for i in range(n) {
        accum = accum + (%s*%s);
    }
    return accum;
}
""" %  (_gen_blir_signature(terms),
        self.lhs.gen_blir_expr(terms), 
        self.rhs.gen_blir_expr(terms))
 
        return term_array, code
        

class Terminal(object):
    def __init__(self, src):
        self.source = src

    # ------------------------------------------------------------
    def __add__(self, rhs):
        return Operation('+', self, rhs)

    def __sub__(self, rhs):
        return Operation('-', self, rhs)

    def __mul__(self, rhs):
        return Operation('*', self, rhs)

    def dot(self, rhs):
        return Operation('dot', self, rhs)

    # ------------------------------------------------------------
    def __repr__(self):
        return 'Terminal(' + repr(self.source) + ')'

    # ------------------------------------------------------------
    def make_terms(self, terms):
        if isinstance(self.source, blaze.Array):
            terms.add(self.source)

    def gen_blir_expr(self, terms):
        if (isinstance(self.source, blaze.Array)):
            return terms[self.source] + '[i]'
        else:
            return repr(self.source)
 
# ----------------------------------------------------------------
def _temp_for(aScalarOrArray, chunk_size):
    if (isinstance(aScalarOrArray, blaze.Array)):
        dtype = aScalarOrArray.datashape.parameters[-1].to_dtype()
        return np.empty((chunk_size,), dtype=dtype)
    else:
        return aScalarOrArray #an Scalar

def _dimension(operand_list):
    dims = [op.datashape.shape[-1].val for op in operand_list if isinstance(op, blaze.Array)]
    assert (dims.count(dims[0]) == len(dims))
    return dims[0]

def chunked_eval(blz_expr, chunk_size=1024):
    operands, code = blz_expr.gen_blir()
    print code
    total_size = _dimension(operands)
    temps = [_temp_for(i, chunk_size) for i in operands]
    temp_op = [i for i in zip(temps, operands) if isinstance(i[1], blaze.Array)]
    offset = 0
    accum = 0.0    
    _, env = blir.compile(code)
    ctx = blir.Context(env)
    while offset < total_size:
        curr_chunk_size = min(total_size - offset, chunk_size)
        slice_chunk = slice(0, curr_chunk_size)
        slice_src = slice(offset, offset+curr_chunk_size)
        for temp, op in temp_op:
            temp[slice_chunk] = op[slice_src]

        accum += blir.execute(ctx, args=temps + [curr_chunk_size], fname='main')
        offset = slice_src.stop

    ctx.destroy()
    return accum


if __name__ == '__main__':
    dshape = '50000000, float64'

    shape, dtype = blaze.to_numpy(blaze.dshape(dshape))
    x = np.ones(shape, dtype=dtype)
    y = np.ones(shape, dtype=dtype)
    z = np.ones(shape, dtype=dtype)
    w = np.ones(shape, dtype=dtype)

    t_np = time()
    result_np = np.dot(x+y, 2.0*z + 2.0*w)
    t_np = time() - t_np

    print 'np result is : %s in %f s' % (result_np, t_np)

    params = blaze.params()
    T = Terminal

    x = T(blaze.ones(dshape, params=params))
    y = T(blaze.ones(dshape, params=params))
    z = T(blaze.ones(dshape, params=params))
    w = T(blaze.ones(dshape, params=params))
    expr = (x+y).dot(T(2.0)*z + T(2.0)*w)

    print expr.gen_blir()[1]

    t_ce = time()
    result_ce = chunked_eval(expr, chunk_size=50000)
    t_ce = time() - t_ce
    print 'ce result is : %s in %f s' % (result_ce, t_ce)


## Local Variables:
## mode: python
## coding: utf-8 
## python-indent: 4
## tab-width: 4
## fill-column: 66
## End:
