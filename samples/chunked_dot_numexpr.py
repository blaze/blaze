"""

Example that evaluates an expression like `(x+y).dot(a*z + b*w)`, where
x, y, z and w are vector that live on disk.  The computation is carried out
by using numexpr.  The code here first massages first the expression above to
achieve something like `sum((x+y)*(a*z + b*w))` that can easily be computed
by numexpr.

Usage:

$ chunked_dot_numexpr create  # creates the vectors on-disk
$ chunked_dot_numexpr run     # computes the expression
$ chunked_dot_numexpr delete  # removes the vectors from disk

"""

import blaze
import numpy as np
import math
from time import time
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


def run_test(args):
    T = Terminal

    x = T('x')
    y = T('y')
    z = T('z')
    w = T('w')
    a = T('a')
    b = T('b')
    evaluator = PythonEvaluator if "python" in args else NumexprEvaluator
    print "evaluating expression with '%s'..." % evaluator.name 
    expr = (x+y).dot(a*z + b*w)

    print 'opening blaze arrays...'
    x_ = blaze.open(_persistent_array_names[0])
    y_ = blaze.open(_persistent_array_names[1])
    z_ = blaze.open(_persistent_array_names[2])
    w_ = blaze.open(_persistent_array_names[3])
    a_ = 2.0
    b_ = 2.0

    if 'in_memory' in args:
        print 'getting an in-memory version of blaze arrays...'
        params = blaze.params(clevel=0)
        t0 = time()
        x_ = blaze.array(x_[:], params=params)
        y_ = blaze.array(y_[:], params=params)
        z_ = blaze.array(z_[:], params=params)
        w_ = blaze.array(w_[:], params=params)
        print "conversion to blaze in-memory: %.3f" % (time() - t0)

    print 'datashape is:', x_.datashape

    if 'print_expr' in args:
        print expr
    
    #warmup
    expr_vars = {'x': x_, 'y': y_, 'z': z_, 'w': w_, 'a': a_, 'b': b_, }
    evaluator(expr, operands=expr_vars).eval() # expr.eval(expr_vars, params={'vm': vm})
    t_ce = time()
    result_ce = evaluator(expr, operands=expr_vars).eval() # expr.eval(expr_vars, params={'vm': vm})
    t_ce = time() - t_ce
    print "'%s' vm result is : %s in %.3f s" % (evaluator.name, result_ce, t_ce)
    
    # in numpy...
    print 'evaluating expression with numpy...'
    x_ = x_[:]
    y_ = y_[:]
    z_ = z_[:]
    w_ = w_[:]

    t_np = time()
    result_np = np.dot(x_+y_, a_*z_ + b_*w_)
    t_np = time() - t_np

    print 'numpy result is : %s in %.3f s' % (result_np, t_np)


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
