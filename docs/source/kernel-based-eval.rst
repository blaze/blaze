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
compiler. In either case, we need the flexibility to
also associate data with each bit of code. For the JIT case
it would be possible to hardcode the shape into the JIT function
as well, but this would prevent reasonable caching of iterators
for similar operands.

In DyND development, an attempt was made to define this
kind of iterator using function pointers and associated data.
This kind of scheme would be the same idea as nditer, but
the layout of the iterator working memory would be
dynamically built up based on the dimension types, and
the 'iternext' would be handled by chaining together function
calls that know how to increment at their level of the
dimension and pass information up as needed.

Handling a variable-sized dimension requires broadcasting its
shape across all the operands during an update step when a
dimension rolls over. This requires manipulating the iterator
shape at that level of the dimensions, based on all the
operand shapes at that level. The effective consequence is
that the nice independence of dimensions and operands
possible when all the operands are strided is gone, and the
snippets of code each type of dimension must provide are
complicated and likely error-prone.

The conclusion within DyND was to discard this avenue of
extending the nditer primitive, and develop the hierarchical
kernel mechanism as the main evaluation mechanism.

Evaluation Based On Hierarchical Kernels
----------------------------------------

To set the stage, I think in blaze we want to keep an
iteration mechanism for one-dimensional iteration and
for simple strided iteration, while adding hierarchical
kernels as the evaluation mechanism which is required to
always work. Iteration is easy to wrap into a hierarchical
kernel, but going the other way would require some
kind of generator/yield mechanism at the C/C++ level.
(e.g. https://github.com/dspeyer/generators)

Hierarchical Assignment Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assignment is a fundamental operation, and one way
to structure computations in blaze would be for every
evaluation to be an assignment to a concrete blaze array.
To illustrate the idea, let's define a simple hierarchical
kernel interface which assigns from python input objects
to numpy arrays.::

    # Simplest kernel
    def scalar_assign_kernel(np_out, any_in, level):
        print("%d scalar_assign_kernel value %s" % (level, any_in))
        np_out[...] = any_in

    # Kernel factory to assign one dimension
    def dimension_kernel_factory(child_kernel):
        def dimension_assign_kernel(np_out, any_in, level):
            print("%d dimension_assign_kernel value %s" % (level, any_in))
            # Check broadcasting rules
            o_len = len(np_out)
            i_len = len(any_in)
            if o_len != i_len and i_len != 1:
                raise RuntimeError('broadcasting error')
            # Do the assignment
            if i_len == 1:
                for i in range(o_len):
                    child_kernel(np_out[i], any_in[0], level + 1)
            else:
                for i in range(o_len):
                    child_kernel(np_out[i], any_in[i], level + 1)
        return dimension_assign_kernel

Now we can build a 2D assignment kernel as follows::

    >>> assign2d = dimension_kernel_factory(
                       dimension_kernel_factory(
                           scalar_assign_kernel))

    # NOTE: Append "1" dimension so numpy doesn't collapse
    #       to scalars.
    >>> a = np.arange(6).reshape(2, 3, 1)
    >>> b = [[5, 6, 7], [8]]
    >>> assign2d(a, b, 0)
    0 dimension_assign_kernel value [[5, 6, 7], [8]]
    1 dimension_assign_kernel value [5, 6, 7]
    2 scalar_assign_kernel value 5
    2 scalar_assign_kernel value 6
    2 scalar_assign_kernel value 7
    1 dimension_assign_kernel value [8]
    2 scalar_assign_kernel value 8
    2 scalar_assign_kernel value 8
    2 scalar_assign_kernel value 8
    >>> print a[..., 0]
    [[5 6 7]
     [8 8 8]]

As you can see, the complexity of the code required to
handle variable-sized array broadcasting is not very high.
The same holds true in a C ABI version of the same
assignment kernel idea, and extends easily to pluggable dimension
types such as chunked, offset, etc.

One reason this is much simpler is that the data required
to manage the state of an individual component kernel is
simply held in local variables, which means on the stack
in the C kernel ABI. This is the natural way to do things
in C/C++, and the equivalent in the iterator approach
described is to build a struct with the needed state, and
accessing those values by casting from a void* to that
struct when the component of the iterator is executed.

Hierarchical Expression Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This idea generalizes to multiple input operands relatively
simply as well. Let's build an addition kernel this way.::

    # Addition kernel
    def addition_kernel(np_out, in0, in1, level):
        print("%d addition values %s %s" % (level, in0, in1))
        np_out[...] = in0 + in1

    # Kernel factory to process one dimension
    def dimension_kernel_factory(child_kernel):
        def dimension_kernel(np_out, in0, in1, level):
            print("%d dimension_kernel values %s %s" % (level, in0, in1))
            # Check broadcasting rules
            o_len = len(np_out)
            i0_len = len(in0)
            i1_len = len(in1)
            if (o_len != i0_len and i0_len != 1) or \
                    (o_len != i1_len and i1_len != 1):
                raise RuntimeError('broadcasting error')
            # Do the assignment
            i0_stride = 0 if i0_len == 1 else 1
            i1_stride = 0 if i1_len == 1 else 1
            for i in range(o_len):
                child_kernel(np_out[i], in0[i*i0_stride],
                                in1[i*i1_stride], level + 1)
        return dimension_kernel

Creating and executing the kernel is basically the same as
for the assignment.::

    >>> assign2d = dimension_kernel_factory(
                           dimension_kernel_factory(
                               addition_kernel))

    # NOTE: Append "1" dimension so numpy doesn't collapse
    #       to scalars.
    >>> a = np.arange(9).reshape(3, 3, 1)
    >>> b = [[5, 6, 7], [8], [9, 10, 11]]
    >>> c = [[-1], [2, 3, 4], [6, 5, 4]]
    >>> assign2d(a, b, c, 0)
    0 dimension_kernel values [[5, 6, 7], [8], [9, 10, 11]] [[-1], [2, 3, 4], [6, 5, 4]]
    1 dimension_kernel values [5, 6, 7] [-1]
    2 addition values 5 -1
    2 addition values 6 -1
    2 addition values 7 -1
    1 dimension_kernel values [8] [2, 3, 4]
    2 addition values 8 2
    2 addition values 8 3
    2 addition values 8 4
    1 dimension_kernel values [9, 10, 11] [6, 5, 4]
    2 addition values 9 6
    2 addition values 10 5
    2 addition values 11 4
    >>> print a[..., 0]
    [[ 4  5  6]
     [10 11 12]
     [15 15 15]]

Handling Fortran/Mixed-order operands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two mechanisms to ensure performance when dealing with Fortran
or mixed memory orders are reordering the iteration axes
(http://docs.scipy.org/doc/numpy/reference/generated/numpy.nditer.html)
and tiling (https://github.com/markflorisson88/minivect/raw/master/thesis/thesis.pdf).

Both of these can be incorporated into the hierarchical
kernel scheme by using appropriate kernel factories. The basic
idea is to have specialized kernel factories that build up
shape and stride information when all the dimensions being
processed are simple strided dimensions, then doing an analysis
of the shape/strides to create an appropriate strided or
tiled kernel.

