====================
Computation Pipeline
====================

This is a developer level document.  It conveys some of the design decisions
around the use of expressions and their lowering to computational backends.  It
is intended for developers.  It is not necessary to understand this document in
order to use Blaze.


Problem
-------

Given an expression:

.. code-block:: python

   x = Symbol('x', '5 * int')
   y = Symbol('y', '5 * int')
   expr = sum(x**2 + y)

And data arranged into a namespace

.. code-block:: python

   xdata = np.array([ 1,  2,  3,  4,  5])
   ydata = np.array([10, 20, 30, 40, 50])

   ns = {x: xdata, y: ydata}

Our goal is to produce the result implied by the expression

.. code-block:: python

   205

Using many small functions defined for each backend to do small pieces of this
computation

.. code-block:: python

   @dispatch(blaze.expr.sum, numpy.ndarray)
   def compute_up(expr, data):
       return numpy.sum(data)

Simple Solution
---------------

A simple solution to this problem is to walk from the leaves of the expression
tree, applying ``compute_up`` functions to data resources until we reach the
top.  In cases like the above example this suffices.  This is called a *bottom
up* traversal.


Complications
-------------

Some backends require more sophistication.  In principle we may want to do the
following:

1.  Modify/optimize the expression tree for a given backend.
    ``optimize(expr, data) -> expr``
2.  Modify the data resources before we start execution.
    ``pre_compute(expr, data) -> data``
3.  Clean up the data result after we complete execution.
    ``post_compute(expr, data) -> data``
4.  Process a leaf of the tree in a bottom up fashion as described above.
    ``compute_up(expr, data) -> data``
5.  Process large chunks of the tree at once, rather than always start from the
    bottom. ``compute_down(expr, data) -> data``

Each of these steps is critical to one backend or another.  We describe each in
turn and then give the complete picture of the entire pipeline.

``optimize :: expr, data -> expr``
---------------------------------

Optimize takes an expression and some data and changes the expression based on
the data type.

For example in columnar stores (like ``bcolz.ctable``) we insert projections in
the expression to reduce the memory footprint.  In numpy-based array backends
we insert ``Broadcast`` operations to perform loop fusion.

This function is applied throughout the tree at the top-most point at which it
is applicable.  It is not applied at leaves which have little to optimize.

``pre_compute :: expr, data -> data``
-------------------------------------

Pre-compute is applied to leaf data elements prior to any computation
(``xdata`` and ``ydata`` in the example above).  It might be used for example,
to load data into memory.

A real use case is the streaming Python backend which consumes either sequences
of tuples or sequences of dicts.  ``precompute(expr, Sequence)`` detects which
case we are in and normalizes to sequences of tuples.  This pre-computation
allows the rest of the Python backend to make useful assumptions.


``post_compute :: expr, data -> data``
--------------------------------------

Post-compute finishes a computation.  It is handed the data after all
computation has been done.

For example, in the case of SQLAlchemy queries the ``post_compute`` function
actually sends the query to the SQL engine and collects results.


``compute_up :: expr, data -> data``
------------------------------------

Compute up walks the expression tree bottom up and processes data step by step.

Compute up is the most prolific function in the computation pipeline and
encodes most of the logic.


``compute_down :: expr, data -> data``
--------------------------------------

In some cases we want to process large chunks of the expression tree at once.
Compute-down operates on the tree top-down, being given the root node / full
expression first, and proceeding down the tree while it can not find a match.

Compute-down is less common than compute-up.  It is most often used when one
backend wants to ship an entire expression over to another.  This is done, for
example, in the SparkSQL backend in which we take the entire expression and
execute it against a SQL backend, and then finally apply that computation onto
the SchemaRDD.

It is also used extensively in backends that leverage chunking.  These backends
want to process a large part of the expression tree at once.


Full Pipeline
-------------

The full pipeline looks like the following

1.  ``Pre-compute`` all leaves of data
2.  ``Optimize`` all appropriate expressions
3.  Start from the top of the expression tree, calling ``compute_down`` until a
    match is found
4.  If no match is found, start from the bottom, calling ``compute_up``
5.  ``Post-Compute`` on the result

This is outlined in ``blaze/compute/core.py`` in the functions ``compute(Expr,
dict)`` and ``top_to_bottom``.
