==============================
Kernel-Based Evaluation Engine
==============================

This is a draft proposal for a way to implement evaluation
using kernels as the primitive and not involving iterators.
This approach still interoperates well with iterators, and
this proposal is not to exclude iterator support in blaze,
but rather suggests kernels as a mechanism which more readily
generalizes to to diversity of data that blaze is targetting.

Dealing With Variable-Sized Dimensions/Ragged Arrays
----------------------------------------------------

The most immediate difficulty with using iterators to
drive the execution engine arises when generalizing
numpy-style broadcasting rules to work with variable-sized
dimensions.

Ragged Array Examples
~~~~~~~~~~~~~~~~~~~~~

Let's work through some simple examples to
illustrate how this works.::

    >>> a = blaze.array([[1, 2], [3]])
    >>> b = blaze.array([[4, 5], [6, 7]])
    >>> print a + b
    blaze.array([[5, 7], [9, 10]])

    >>> a = blaze.array([[1, 2], [3]])
    >>> b = blaze.array([[4], [5, 6, 7]])
    >>> print a + b
    blaze.array([[5, 6], [8, 9, 10]])

    >>> a = blaze.array([[[1, 2, 3], [4, 5]], [[6, 7, 8, 9]]])
    >>> print a.sum(axis=2)
    blaze.array([[6, 9], [30]])

    >>> a = blaze.array([[[1, 2], [3, 4]], [[5, 6]]])
    >>> print a.sum(axis=1)
    blaze.array([[4, 6], [5, 6]])

A simple application of this is calculating aggregates on
groupby results. The groupby operation results in a
deferred 2D ragged array, and aggregates taken along
the second axis are a typical step immediately after
the groupby.::

    >>> gb = blaze.groupby(a, by=a.state)
    >>> cars_per = gb.sum(a.car_counts, axis=1) / state_populations

Review of Strided Array Iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numpy-style arrays have a simple multidimensional structure
which is very amenable to iteration with broadcasting. This
is the basis of numpy's nditer object. Given a number of
operand arrays, the result of broadcasting them all together
gives an iteration space. Let's work through an example
where the operands have shapes (3, 1, 4), (5, 1), (4).
Broadcasting the shapes together makes::

    (3, 1, 4) # A
       (5, 1) # B
          (4) # C
    ---------
    (3, 5, 4) # Iteration space

Each array can be transformed into an array whose shape matches
the iteration space by using zero strides. All dimensions which
are missing or have size one have their stride set to zero.::

    A.strides == (16, 16, 4)
    Aprime.strides == (16, 0, 4)
    B.strides == (8, 40)
    Bprime.strides == (0, 8, 0)
    C.strides == (4)
    Cprime.strides == (0, 0, 4)

Now with Aprime, Bprime, and Cprime having exactly the
same shape, it is easy to do iteration across all of
them at the same time. This is numpy's multi-iterator,
used before nditer, and nditer work. The multi-iterator
does this very explicitly, creating one iterator for each
operand, while nditer creates a more cache-friendly
structure holding all the dimensions sizes, strides, and
iteration indexes, so it can do this iteration
more efficiently. In particular, each iteration
dimension gets a structure roughly like::

    struct nditer_dimension {
        intptr_t idx, size;
        const void *ptr[N];
        intptr_t stride[N];
    };

Because all the operands have the same iteration space,
only one iteration multi-index/shape needs to be tracked.
The update step for the pointers can also be vectorized
by padding N to a multiple of the vector size, though
the numpy nditer does not do this.

This structure makes it easy to reorder the
axes, and coalesce them where such operations might
improve performance. If the operands are suitably
nice contiguous arrays, this can result in multi-dimensional
iterations transforming into a one-dimensional iteration.
NDIter also provides an option for the user of the
iterator to handle the innermost iteration dimension.

Generalizing To Variable-sized Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When any dimension of the operands could also be
a variable-sized dimension, all of the nice structure
of this iteration goes away. First, the
iteration space may be ragged, so it can't be
characterized by an index and stride per dimension.
Since each dimension after the first could be variable
or not, there are 2^(N-1) possibilities for how
variable/strided dimensions may be chosen.

There are a few ways to deal with this combinatorial
explosion of possibilities. One way is to switch to
fully dynamic data per dimension, and using a
switch statement or virtual function call to dispatch
each iteration step. This loses much of the array advantages,
taking us much closer to CPython performance characteristics.

Another way is to assemble bits of code tuned to the
dimension types, either via function pointers or with a JIT
compiler.

... (WIP)
