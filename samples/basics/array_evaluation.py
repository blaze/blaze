'''Sample script showing the way to perform computations in blaze'''

from __future__ import print_function
#
# A very simple script showing evaluation
#
# This should be executable and result in an out of core execution to
# generate the result of the expression
#
# This illustrates the idea of:
#
# - Using large in-disk arrays as operands
#
# - Building expressions to evaluate in blaze
#
# - Evaluate those expressions to produce a result
#
#   - Showing that we can configure how we evaluate expressions
#
#   - Showing how we can specify the kind of result we desire

import blaze

def generate_operand(uri):
    """build some operands on disk"""
    pass

def evaluation(operand_dict):
    a = blaze.load(operand_dict['a'])
    b = blaze.load(operand_dict['b'])

    expr = (a+b)*(a*b)

    print(type(expr)) # would this be "blaze.array"?
    print(type(expr._data)) # would this be blaze.BlazeFuncDataDescriptor?

    print(expr) # what should this print???
    c = blaze.eval(expr, out_caps={}, hints={})
    print(c) #should print the result... rejoice!

if __name__ == '___main___':
    sys.exit(main(sys.argv)
