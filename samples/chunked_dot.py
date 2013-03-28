""" Sample showing chunked execution of expresions

This sample constructs an expresion to be executed in chunks.
Different aproaches are tested, and it is compared with the
equivalent expresion written in numpy.
"""

import blaze
import numpy as np
from time import time
import blaze
from chunked import Operation, Terminal, BlirEvaluator, NumexprEvaluator, NumpyEvaluator

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


def make_expression(in_memory=False):
    T = Terminal

    expr = (T('x')+T('y')).dot(T('a')*T('z') + T('b')*T('w'))
    print 'opening blaze arrays...'
    arrays = map(blaze.open, _persistent_array_names)

    if (in_memory):
        print 'getting an in-memory version of blaze arrays...'
        t0 = time()
        params = blaze.params(clevel=9)
        arrays = [blaze.array(array, params=params) for array in arrays]
        print "conversion to blaze in-memory: %.3f" % (time() - t0)

    print 'datashape is:', arrays[0].datashape

    return expr, { 'x': arrays[0], 
                   'y': arrays[1], 
                   'z': arrays[2], 
                   'w': arrays[3], 
                   'a': 2.0, 
                   'b': 2.0 }


def run_test(args):
    expr, operands = make_expression(in_memory='in_memory' in args)

    t_bc = time()

    if 'blir' in args:
        evaluator = BlirEvaluator
    elif 'numexpr' in args:
        evaluator = NumexprEvaluator
    else:
        evaluator = NumpyEvaluator

    evaluator = evaluator(expr, operands=operands)
    t_bc = time() - t_bc
    print '%s took %f s to build' % (evaluator.name, t_bc)

    for log2cs in xrange(12, 26):
        evaluator.time = 0.0
        cs = 2**log2cs
        t_ce = time()
        result_ce = evaluator.eval(cs)
        t_ce = time() - t_ce
        print '%s result is : %s in %f/%f s (chunksize = %d)' % (evaluator.name, result_ce, t_ce, evaluator.time, cs)

    # in numpy...
    t0 = time()
    x = operands['x'][:]
    y = operands['y'][:]
    z = operands['z'][:]
    w = operands['w'][:]
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
        print args[0] + ' [create <elements>|run|delete]'


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
