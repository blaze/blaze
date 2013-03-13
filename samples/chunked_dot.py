import blaze
import blaze.blir as blir
import numpy as np
from time import time
import blaze


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

class Operation(object):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __add__(self, rhs):
        return Operation('+', self, rhs)

    def __sub__(self, rhs):
        return Operation('-', self, rhs)

    def __mul__(self, rhs):
        return Operation('*', self, rhs)

    def dot(self, rhs):
        return Operation('dot', self, rhs)

    def make_terms(self, terms):
        self.lhs.make_terms(terms)
        self.rhs.make_terms(terms)
        return terms

    def __repr__(self):
        return ('Operation(' + repr(self.op) + ', ' 
                + repr(self.lhs) + ', ' 
                + repr(self.rhs) + ')')

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

    def __add__(self, rhs):
        return Operation('+', self, rhs)

    def __sub__(self, rhs):
        return Operation('-', self, rhs)

    def __mul__(self, rhs):
        return Operation('*', self, rhs)

    def dot(self, rhs):
        return Operation('dot', self, rhs)

    def make_terms(self, terms):
        terms.add(self.source)

    def gen_blir_expr(self, terms):
        return terms[self.source] + '[i]'
 
    def __repr__(self):
        return 'Terminal(' + repr(self.source) + ')'

_src = """
def main(x: array[float], y: array[float], n : int) -> float {
    var float accum = 0.0;
    var int i = 0;

    for i in range(n) {
        accum = accum + x[i]*y[i];
    }
    return accum;
}
"""
_dot_ast, _dot_env = blir.compile(_src)


def chunked_dot(a, b, chunk_size=1024):
    a_shape, a_dtype = blaze.to_numpy(a.datashape)
    b_shape, b_dtype = blaze.to_numpy(b.datashape)
    assert(a_dtype == b_dtype)
    assert(a_dtype == np.float64)
    assert(len(a_shape) == 1)
    assert(len(b_shape) == 1)
    assert(a_shape[0] == b_shape[0])

    achunk = np.empty((chunk_size,), a_dtype)
    bchunk = np.empty((chunk_size,), b_dtype)
    total_size = a_shape[0]
    accum = 0.0;
    offset = 0;

    while offset < total_size:
        t0 = time()
        t1 = time()
        curr_chunk_size = min(total_size - offset, chunk_size)
        slice_chunk = slice(0, curr_chunk_size)
        slice_src = slice(offset, offset+curr_chunk_size)
        achunk[slice_chunk] = a[slice_src]
        bchunk[slice_chunk] = b[slice_src]
        t2 = time()
        accum += np.dot(achunk[slice_chunk], bchunk[slice_chunk])
#blir.execute(_dot_env, (achunk, bchunk, curr_chunk_size))
        t3 = time()
        offset = slice_src.stop
        print 'chunk at %d: compile %f, copy %f, exe %f' % (offset, t1-t0,t2-t1,t3-t2)
        
    return accum

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

        accum += blir.execute(ctx, args=temps + [curr_chunk_size])
        offset = slice_src.stop

    return accum


if __name__ == '__main__':
    dshape = '1000000, float64'
    params = blaze.params()
    x = Terminal(blaze.ones(dshape, params=params))
    y = Terminal(blaze.ones(dshape, params=params))
    z = Terminal(blaze.ones(dshape, params=params))
    w = Terminal(blaze.ones(dshape, params=params))
    a = Terminal(blaze.ones(dshape, params=params))
    b = Terminal(blaze.ones(dshape, params=params))
    expr = (x+y).dot(a*z + b*w)

    print expr.gen_blir()[1]

    t_ce = time()
    result1 = chunked_eval(expr, chunk_size=50000)
    t_ce = time() - t_ce
    print 'ce result is : %s in %f s' % (result1, t_ce)

    shape, dtype = blaze.to_numpy(blaze.dshape(dshape))
    x = np.ones(shape, dtype=dtype)
    y = np.ones(shape, dtype=dtype)
    z = np.ones(shape, dtype=dtype)
    w = np.ones(shape, dtype=dtype)
    a = np.ones(shape, dtype=dtype)
    b = np.ones(shape, dtype=dtype)

    t_np = time()
    result = np.dot(x+y, a*z + b*w)
    t_np = time() - t_np

    print 'np result is : %s in %f s' % (result1, t_np)
