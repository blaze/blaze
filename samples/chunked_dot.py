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
from chunked.expression_builder import Operation, Terminal, Visitor


# ================================================================

class BlirEvaluator(object):
    """Evaluates expressions using blir

    Note that this is a sample, and has several assumptions
    hardcoded. This is 'proof-of-concept'
    """

    class _ExtractTerminals(Visitor):
        def accept_operation(self, node):
            return self.accept(node.lhs) |  self.accept(node.rhs)
           
        def accept_terminal(self, node):
            if isinstance(node.source, blaze.Array):
                return { node.source }
            else:
                return set()

    class _GenerateExpression(Visitor):
        def __init__(self, terms_dict):
            self.terms = terms_dict

        def accept_operation(self, node):
            a = self.accept(node.lhs)
            b = self.accept(node.rhs)
            return '(' + a + node.op + b + ')'
                
        def accept_terminal(self, node):
            if (isinstance(node.source, blaze.Array)):
                return self.terms[node.source] + '[i]'
            else:
                return repr(node.source)

    def __init__(self, root_node):
        """ after constructor:
        """
        assert(root_node.op == 'dot')
        terms = self._ExtractTerminals().accept(root_node)
        terms = { obj: 'in%d' % i for i, obj in
                  enumerate(terms) }

        str_signature = self._gen_blir_signature(terms)
        str_lhs = self._GenerateExpression(terms).accept(root_node.lhs)
        str_rhs = self._GenerateExpression(terms).accept(root_node.rhs)
        code = """
def main(%s, n: int) -> float {
    var float accum = 0.0;
    var int i = 0;
    for i in range(n) {
        accum = accum + (%s*%s);
    }
    return accum;
}
""" %  (str_signature, str_lhs, str_rhs)

        _, self.env = blir.compile(code)
        self.ctx = blir.Context(self.env)
        self.operands = list(terms)
        self.code = code

    def __del__(self):
        try:
            self.ctx.destroy()
        except:
            pass

    def chunked_eval(self, chunk_size = 32768):
        operands = self.operands
        total_size = operands[0].datashape.shape[-1].val
        offset = 0
        accum = 0.0
        t_real = 0.0
        while offset < total_size:
            curr_chunk_size = min(total_size - offset, chunk_size)
            slice_src = slice(offset, offset+curr_chunk_size)
            args = [op[slice_src] for op in operands]
            args.append(curr_chunk_size)
            t = time()
            accum += blir.execute(self.ctx, args=args, fname='main')
            t_real += time() - t
            offset = slice_src.stop

        return accum, t_real

    
    def __str__(self):
        str_args = ',\n\n'.join([str(op) for op in  self.operands]) 
        return '''blir evaluator with code:\n%s\n\nwith args:\n%s\n''' % (self.code, str_args)

    @staticmethod
    def _to_blir_type_string(obj):
        if (isinstance(obj, blaze.Array)):
            p = obj.datashape.parameters
            assert(len(p) == 2)
            return 'array[%s]' % 'float'
        else:
            return 'float'

    @staticmethod
    def _gen_blir_decl(name, obj):
        return name + ': ' + BlirEvaluator._to_blir_type_string(obj)

    @staticmethod
    def _gen_blir_signature(terms):
        return ',\n\t'.join([BlirEvaluator._gen_blir_decl(pair[1], pair[0])
                             for pair in terms.iteritems()])

# ================================================================

_persistent_array_names = ['chunk_sample_x.blz',
                           'chunk_sample_y.blz',
                           'chunk_sample_z.blz',
                           'chunk_sample_w.blz']


def _create_persistent_array(name, dshape):
    print 'creating ' + name + '...'
    blaze.ones(dshape, params=blaze.params(storage=name, clevel=0))


def _delete_persistent_array(name):
    from shutil import rmtree
    rmtree(name)


def create_persistent_arrays(args):
    elements = args[0] if len(args) > 0 else '10000000'
    dshape = elements + ', float64'

    try:
        dshape = blaze.dshape(dshape)
    except:
        print elements + ' is not a valid size for the arrays'
        return

    for name in _persistent_array_names:
        _create_persistent_array(name, dshape)


def delete_persistent_arrays():
    for name in _persistent_array_names:
        _delete_persistent_array(name)


def run_test(args):
    T = Terminal

    print 'opening blaze arrays...'
    x = blaze.open(_persistent_array_names[0])
    y = blaze.open(_persistent_array_names[1])
    z = blaze.open(_persistent_array_names[2])
    w = blaze.open(_persistent_array_names[3])

    if 'in_memory' in args:
        print 'getting an in-memory version of blaze arrays...'
        params = blaze.params(clevel=9)
        t0 = time()
        x = blaze.array(x, params=params)
        y = blaze.array(y, params=params)
        z = blaze.array(z, params=params)
        w = blaze.array(w, params=params)
        print "conversion to blaze in-memory: %.3f" % (time() - t0)

    print 'datashape is:', x.datashape

    print 'evaluating expression with blir...'
    expr = (T(x)+T(y)).dot(T(2.0)*T(z) + T(2.0)*T(w))

    if 'print_expr' in args:
        print expr.gen_blir()[1]

    t_bc = time()
    evaluator = BlirEvaluator(expr)
    t_bc = time() - t_bc
    print 'blir evaluator took %f s to build' % t_bc

    for log2cs in xrange(12, 26):
        cs = 2**log2cs
        t_ce = time()
        result_ce, t_real = evaluator.chunked_eval(cs)
        t_ce = time() - t_ce
        print 'blir chunked result is : %s in %f/%f s (chunksize = %d)' % (result_ce, t_ce, t_real, cs)

    # in numpy...
    t0 = time()
    x = x[:]
    y = y[:]
    z = z[:]
    w = w[:]
    print "Conversion to numpy in-memory: %.3f" % (time() - t0)

    print 'evaluating expression with numpy...'
    t_np = time()
    result_np = np.dot(x+y, 2.0*z + 2.0*w)
    t_np = time() - t_np

    print 'numpy result is : %s in %f s' % (result_np, t_np)


def main(args):
    command = args[1] if len(args) > 1 else 'help'

    if command == 'create':
        create_persistent_arrays(args[2:])
    elif command == 'run':
        run_test(args)
    elif command == 'delete':
        delete_persistent_arrays()
    else:
        print args[0] + ' [create elements|run|delete]'


if __name__ == '__main__':
    from sys import argv
    main(argv)


## Local Variables:
## mode: python
## coding: utf-8
## python-indent: 4
## tab-width: 4
## fill-column: 66
## End:
