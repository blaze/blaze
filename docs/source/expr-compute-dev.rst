===========================
Expressions and Computation
===========================

This is a developer level document.  It conveys some of the design decisions
around the use of expressions and their lowering to computational backends.  It
is intended for new developers.  It is not necessary to understand this
document in order to use Blaze.

Expressions
-----------

Blaze represents expressions as Python objects.  Classes include

- **Symbol**: leaf expression, ``t``
- **Projection**: subset of columns, ``t[['name', 'amount']]``
- **Selection**: subset of rows ``t[t.amount < 0]``
- **Field**: single column of data or field of record dataset ``t.name``
- **Broadcast**: a scalar expression broadcast to a collection, ``t.amount + 1``
- **Join**: join two expressions on shared fields , ``join(t, s, 'id')``
- **Reduction**: perform a sum or min or max on a collection, ``t.amount.sum()``
- **By**: split-apply-combine operation, ``by(t.name, total=t.amount.sum())``
- **Also**: ``Sort, Distinct, Head, Label, Map, Merge, ...``

In each case an operation (like ``Selection``) is a Python class.  Each
expression defines a fixed set of fields in the ``__slots__`` attribute

.. code-block:: python

   class Selection(Expr):
       __slots__ = '_hash', '_child', 'predicate'

   class Field(ElemWise):
       __slots__ = '_hash', '_child', 'fieldname'


To create a node in the tree explicitly we create a Python object of this class

.. code-block:: python

   >>> from blaze.expr import *
   >>> t = Symbol('t', 'var * {id: int, name: string, amount: int}')
   >>> amounts = Field(t, 'amount')

This object contains its information in a .args attribute

.. code-block:: python

   >>> amounts._args == (t, 'amount')
   True

And the set of input expressions in a ``._inputs`` attribute

.. code-block:: python

   >>> amounts._inputs == (t,)
   True

By traversing ``._args`` one can traverse the tree of all identifying
information (including annotating strings and values like ``'amount'``) or by
traversing ``._inputs`` one can inspect the much sparser tree of just the major
expressions, skipping parameters like the particular field name to be
selected.

Most terms have only a single child input.  And so often the ``._inputs`` tree
is just a single line of nodes.  Notable exceptions include operations like
``Join`` and ``BinOp`` which contain two inputs.


Expression Invariants
---------------------

Blaze expressions adhere to the following properties:

1.  They and all of their stored fields are immutable
2.  Their string representations evaluate to themselves.  E.g.
    ``eval(str(expr)) == expr``
3.  They have simple ``__init__`` constructors that only copy in fields to the
    object.  For intelligent argument handling they have functions.  E.g. the
    ``Join`` class has an analagous ``join`` function that should be used by
    users.  Same with the internal ``By`` class as the user-level ``by``
    function.
4.  They can compute their datashape ``.dshape`` given the datashape of their
    children and their arguments.


Organization
------------

All expr code occurs in ``blaze/expr/``.  This directory should be
self-contained and not dependent on other parts of Blaze like ``compute`` or
``api``.

* ``blaze/expr/core.py`` contains code related to abstract tree traversal
* ``blaze/expr/expr.py`` contains code related to datashape imbued expressions
* ``blaze/expr/collections.py`` contains operations related to expressions with
  datashapes that contain a dimension.  Operations like ``Selection`` and
  ``Join`` live here
* ``blaze/expr/datetime.py``, ``blaze/expr/string.py``, ...  all contain
  specialized operations for particular domains.

Computation
-----------

Once we have a Blaze expression like the following:

.. code-block:: python

   >>> deadbeats = t[t.amount < 0].name

and some data like the following:

.. code-block:: python

   >>> data = [[1, 'Alice', 100],
   ...         [2, 'Bob', -200],
   ...         [3, 'Charlie', 300]]

and a mapping of Symbols to data like the following:

.. code-block:: python

   >>> namespace = {t: data}

then we need to evaluate the intent of the expression on the data.  We do this
in a step-by-step system outlined by various ``compute`` functions.  The user
experience is as follows

.. code-block:: python

   >>> from blaze import compute
   >>> list(compute(deadbeats, namespace))
   ['Bob']

But internally ``compute`` traverses our expression from the leaves (like
``t``) on up, transforming ``data`` as it goes.  At each step it looks at a
node in the Blaze expression graph like the following:

.. code-block:: python

   >>> selection_t = t[t.amount < 0]

and transforms the data appropriately, like the following:

.. code-block:: python

   >>> predicate = lambda amt: amt < 0
   >>> data = filter(predicate, data)

This step-by-step approach is easy to define through dispatched ``compute_up``
functions.  We create a small recipe for how to compute each expression type
(e.g. ``Projection``, ``Selection``, ``By``) against each data type (e.g.,
``list``, ``DataFrame``, ``sqlalchemy.Table``, ....)  Here is the recipe
mapping a ``Selection`` to a ``DataFrame``:

.. code-block:: python

   >>> @dispatch(Selection, DataFrame)   # doctest: +SKIP
   ... def compute_up(t, df, **kwargs):
   ...     predicate = compute(t.predicate, df)
   ...     return df[predicate]

This approach is modular and allows interpretation systems to be built up as a
collection of small pieces.  One can begin the construction of a new backend by
showing Blaze how to perform each individual operation on a new data type.  For
example here is a start of a backend for PyTables:

.. code-block:: python

   >>> @dispatch(Selection, tb.Table)    # doctest: +SKIP
   ... def compute_up(expr, data):
   ...     s = eval_str(expr.predicate)  # Produce string like 'amount < 0'
   ...     return data.read_where(s)     # Use PyTables read_where method

   >>> @dispatch(Head, tb.Table)         # doctest: +SKIP
   ... def compute_up(expr, data):
   ...     return data[:expr.n]          # PyTables supports standard indexing

   >>> @dispatch(Field, tb.Table)       # doctest: +SKIP
   ... def compute_up(expr, data):
   ...     return data.col(expr._name)  # Use the PyTables .col method


These small functions are isolated enough from Blaze to be easy for new
developers to write, even without deep knowledge of Blaze internals.


Compute Traversal
-----------------

The ``compute_up`` functions expect to be given:

1.  The expression containing information about the computation to be performed
2.  The data elements corresponding to the ``.inputs`` of that expression

The ``compute`` function orchestrates ``compute_up`` functions and performs
the actual traversal, accruing intermediate results from the use of
``compute_up``.  By default ``compute`` performs a ``bottom_up`` traversal.
First it evaluates the leaves of the computation by swapping out keys for
values in the input dictionary, ``{t: data}``.  It then calls ``compute_up``
functions on these leaves to find intermediate nodes in the tree.  It repeats
this process, walking up the tree, and at each stage translating a Blaze
expression into the matching data element given the data elements of the
expression's children.  It continues this process until it reaches the root
node, at which point it can return the result to the user.

Sometimes we want to perform pre-processing or post-processing on the
expression or the result.  For example when calling ``compute`` on a
``blaze.data.SQL`` object we actually want to pre-process this input to extract
out the ``sqlalchemy.Table`` object and call ``compute_up`` on that.  When
we're finished and have successfully translated our Blaze expression to a
SQLAlchemy expression we want to post-process this result by actually running
the query in our SQL database and returning the concrete results.



Adding Expressions
------------------

Expressions can be added by creating a new subclass of
:class:`blaze.expr.expressions.Expr`. When adding a class, one should define all
of the instance data that the type will need in the ``__slots__``. This should
probably include a ``_hash``. which will be used to cache the hash value of the
node. Often we should defer to the ``__init__`` defined on the super class. This
will reflect the signature from the ``__slots__``.

To define the shape of our new expression, we should implement the
``_dshape`` method. This method should use the shapes of the arguments
passed in the constructor plus knowledge of this type of transformation to
return the datashape of this expression. For example, thinking of ``sum``, we
would probably want a method like:

.. code-block:: python

   def _dshape(self):
       # Drop the dimension of the child reducing to a scalar type.
       return self._child.schema.measure


Here we see the ``.schema`` attribute being used. This attribute dispatches to
another optional method: ``_schema``. This method should return the datashape
with the shape stripped off, or just the data type. If this is not defined, it
will be implemented in terms of the ``_dshape`` method. This is often convenient
for subclasses where the rules about the ``schema`` change but the rules for the
dimensions are all the same, like :class:`blaze.expr.reductions.Reduction`.

The constructor is not public construction point for a blaze expression. After
the class is defined a pairing function should be added to construct and type
check the new node. For example, if our node is ``Concat``, then the functions
should be called ``concat``. We will want to decorate this function with
:func:`odo.utils.copydoc` to pull the docstring from the class. This function's
main job is type checking the operands. Any constructed node should be in a
valid state. If the types do not check out, simply raise a ``TypeError`` with a
helpful message to the user.

Now that the new expression class is defined and the types work out, it must be
dispatched to in the compute backends. For each backend that can implement this
new feature, a corrosponding ``compute_up`` dispatch should be defined. For
example, assuming we just defined ``sum``, we would need to implement something
like:

.. code-block:: python

   @dispatch(sum, np.ndarray)
   def compute_up(expr, arr, **kwargs):
       ...

   @dispatch(sum, pd.Series)
   def compute_up(expr, arr, **kwargs):
       ...

   @dispatch(sum, (list, tuple))
   def compute_up(expr, arr, **kwargs):
       ...

   ...


Each of these function definitions should appear in the ``blaze.compute.*``
module for the given backend. For example, the ``ndarray`` definition should go
in ``blaze.compute.numpy``.

After implementing the various compute up functions, tests should be written for
this behavior. Tests should be added to ``blaze/expr/tests`` for the expression
itself, including tests against the construction and the dshape. Tests are also
needed for each of the particular backend implementations to assert that the
results of performing the computation is correct accross our various backends.
