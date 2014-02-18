# -*- coding: utf-8 -*-

"""
This guide will hopefully shed some light to how blaze functions can be
defined and implemented.

The idea is that blaze has a reference implementation for each operation it
defines in Python. Of course, others are free to add operations specific
to their problem domain, and have them participate in the blaze deferred
execution strategy.

Our goal is open-ended extension, of existing and new functions blaze knows
nothing about. The way this is currently realized is by defining a blaze
function. This function contains:

    * a reference implementation, callable from Python

        This has a type signature and may implement the operation or simply
        raise an exception

            @blaze_func('A -> A -> A', elementwise=True)
            def add(a, b):
                return a + b # We add the scalar elements together

        Here we classified the function as `elementwise`, which gets scalar
        inputs. Generally, blaze functions can be regarded as generalized
        ufuncs, and their inputs are the array dimensions they matched
        according to their signature.

        The 'A -> A -> A' part is the signature, which indicates a function
        taking two arguments of a compatible type (`A` and `A`) and returning
        another `A`. This means that the
        argument types must be the compatible, and arguments are subject to
        promotion. For instance, if we put in an (int, float), the system will
        automatically promote the int to a float. Note that the types in the
        signature, in this case the type variable `A`, are identified by
        position.

    * a set of overloaded implementations of certain implementation "kinds"

        e.g. we may have for the blaze.add function the following
        implementations:

            ckernel:

                overloaded ckernel for each type, e.g.

                    int32 ckernel_add(int32 a, int32 b) { return a + b; }
                    float32 ckernel_add(float32 a, float32 b) { return a + b; }

            sql:

                 overloaded sql implementation, this can be generic for
                 all numeric input types, e.g.:

                 @impl(blaze.add, 'a : numeric -> a -> a')
                 def sql_add(a, b):
                    # using dumb string interpolation to generate an SQL
                    # query
                    return "(%s + %s)" % (a, b)

The expression graph is then converted to AIR (the Array Intermediate
Representation), which is processed in a number of passes to:

    1) handle coercions
    2) assign "implementation backends" to each operation, depending on

        * location of data (in-memory, on-disk, distributed, silo, etc)
        * availability of implementation "kernels" (e.g. ckernel, sql, etc)

    This is done in a straightforward, greedy, best-effort fashion. There is
    no cost model to guide this process, we only try to mimimize data transfer.


When blaze functions are applied, it build an expression graph automatically,
which refers to the blaze function that was applied and the arguments it was
applied with. It performs type unification on the signatures of the reference
(python) implementation. This specifies the semantics of the blaze function,
and under composition the semantics of the entire blaze execution system. The
added benefit of having a pure-python reference implementation is that we can
run the execution system in "debug" mode which uses python to evaluate a blaze
expression.
"""

from __future__ import absolute_import, division, print_function
from itertools import cycle

import blaze
from blaze.compute.function import blaze_func, function, kernel

def broadcast_zip(a, b):
    """broadcasting zip"""
    assert len(a) == len(b) or len(a) == 1 or len(b) == 1
    n = max(len(a), len(b))
    for _, x, y in zip(range(n), cycle(a), cycle(b)):
        yield x, y


@function('axes..., axis, dtype -> axes..., axis, bool -> axes..., var, dtype')
def filter(array, conditions):
    """
    Filter elements from `array` according to `conditions`.

    This is the reference function that dictates the semantics, input and
    output types of any filter operation. The function may be satisfied by
    subsequent (faster) implementations.

    Example
    =======

    >>> filter([[1, 2],        [3, 4],       [5, 6]],
    ...        [[True, False], [True, True], [False, False]])
    [[1], [3, 4], []]
    """
    return py_filter(array, conditions, array.ndim)


def py_filter(array, conditions, dim):
    """Reference filter implementation in Python"""
    if dim == 1:
        result = [item for item, cond in bzip(array, conditions) if cond]
    else:
        result = []
        for item_subarr, conditions_subarr in bzip(array, conditions):
            result.append(py_filter(item_subarr, conditions_subarr, dim - 1))

    return result


def ooc_filter(array, conditions):
    pass


if __name__ == '__main__':
    #import numpy as np
    from dynd import nd

    arr = nd.array([[1, 2], [3, 4], [5, 6]])
    conditions = nd.array([[True, False], [True, True], [False, False]])
    expr = filter(arr, conditions)

    result = blaze.eval(expr, strategy='py')
    print(">>> result = filter([[1, 2], [3, 4], [5, 6]],\n"
          "                    [[True, False], [True, True], [False, False]])")
    print(">>> result")
    print(result)
    print(">>> type(result)")
    print(type(result))

    expr = filter(arr + 10, conditions)
    result = blaze.eval(expr, strategy='py')
    print(">>> result = filter([[1, 2], [3, 4], [5, 6]] + 10,\n"
          "                    [[True, False], [True, True], [False, False]])")
    print(">>> result")
    print(result)
    print(">>> type(result)")
    print(type(result))
