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

   >>> from blaze import symbol, sum
   >>> x = symbol('x', '5 * int')
   >>> y = symbol('y', '5 * int')
   >>> expr = sum(x ** 2 + y)
   >>> expr
   sum((x ** 2) + y)

And data arranged into a namespace

.. code-block:: python

   >>> import numpy as np
   >>> xdata = np.array([1, 2, 3, 4, 5])
   >>> ydata = np.array([10, 20, 30, 40, 50])
   >>> ns = {x: xdata, y: ydata}

Our goal is to produce the result implied by the expression

.. code-block:: python

   >>> np.sum(xdata ** 2 + ydata)
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
3.  Modify the data resources as they change type throughout the computation
    ``pre_compute(expr, data) -> data``
4.  Clean up the data result after we complete execution.
    ``post_compute(expr, data) -> data``
5.  Process a leaf of the tree in a bottom up fashion as described above.
    ``compute_up(expr, data) -> data``
6.  Process large chunks of the tree at once, rather than always start from the
    bottom. ``compute_down(expr, data) -> data``

Each of these steps is critical to one backend or another.  We describe each in
turn and then give the complete picture of the entire pipeline.

``optimize :: expr, data -> expr``
----------------------------------

Optimize takes an expression and some data and changes the expression based on
the data type.

For example in columnar stores (like ``bcolz.ctable``) we insert projections in
the expression to reduce the memory footprint.  In numpy-based array backends
we insert ``Broadcast`` operations to perform loop fusion.

This function is applied throughout the tree at the top-most point at which it
is applicable.  It is not applied at leaves which have little to optimize.

``pre_compute :: expr, data -> data``
-------------------------------------

Pre-compute is applied to leaf data elements prior to computation
(``xdata`` and ``ydata`` in the example above).  It might be used for example,
to load data into memory.

We apply ``pre_compute`` at two stages of the pipeline

1.  At the beginning of the computation
2.  Any time that the data significantly *changes type*

So for example for the dataset::

   data = {'my_foo':  Foo(...)}

If we apply the computation::

   X -> X.my_foo.distinct()

Then after the ``X -> X.my_foo`` computation as the type changes from ``dict``
to ``Foo`` we will call ``pre_compute`` again on the ``Foo`` object with the
remaining expression::

    data = pre_compute(X.my_foo.distinct(), Foo(...))

A real use case is the streaming Python backend which consumes either sequences
of tuples or sequences of dicts.  ``precompute(expr, Sequence)`` detects which
case we are in and normalizes to sequences of tuples.  This pre-computation
allows the rest of the Python backend to make useful assumptions.

Another use case is computation on CSV files.  If the CSV file is small we'd
like to transform it into a pandas DataFrame.  If it is large we'd like to
transform it into a Python iterator.  This logic can be encoded as a
``pre_compute`` function and so will be triggered whenever a ``CSV`` object is
first found.


``post_compute :: expr, data -> data``
--------------------------------------

Post-compute finishes a computation.  It is handed the data after all
computation has been done.

For example, in the case of SQLAlchemy queries the ``post_compute`` function
actually sends the query to the SQL engine and collects results.  This occurs
only after Blaze finishes translating everything.


``compute_up :: expr, data -> data``
------------------------------------

Compute up walks the expression tree bottom up and processes data step by step.

Compute up is the most prolific function in the computation pipeline and
encodes most of the logic.  A brief example

.. code-block:: python

   @dispatch(blaze.expr.Add, np.ndarray, np.ndarray)
   def compute_up(expr, lhs, rhs):
       return lhs + rhs


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
2.  ``Optimize`` the expression
3.  Try calling ``compute_down`` on the entire expression tree
4.  Otherwise, traverse up the tree from the leaves, calling ``compute_up``.
    Repeat this until the data significantly changes type (e.g. ``list`` to
    ``int`` after a ``sum`` operation)
5.  Reevaluate ``optimize`` on the expression and ``pre_compute`` on all of the
    data elements.
6.  Go to step 3
7.  Call ``post_compute`` on the result

This is outlined in ``blaze/compute/core.py`` in the functions ``compute(Expr,
dict)`` and ``top_then_bottom_then_top_again_etc``.


History
-------

This design is ad-hoc.  Each of the stages listed above arose from need, not
from principled fore-thought.  Undoubtedly this system could be improved.  In
particular much of the complexity comes from the fact that ``compute_up/down``
functions may transform our data arbitrarily.  This, along with various
particular needs from all of the different data types, forces the
flip-flopping between top-down and bottom-up traversals.  Please note that
while this strategy *works well most of the time* pathalogical cases do exist.
