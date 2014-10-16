
=======================
Interactive Expressions
=======================

Internally Blaze is abstract; this limits interactivity.  Blaze *interactive
expressions* resolve this issue and provide a smooth experience to handling
foreign data.

Expressions with Data
---------------------

Internally Blaze separates the intent of the computation from the data/backend.
While powerful, this abstract separation limits interactivity, one of the
core goals of Blaze.

Blaze *interactive expressions* are like normal expressions but their leaves
may hold on to a concrete data resource (like a DataFrame or SQL database.)
This embeds a specific data context, providing user interface improvements at
the cost of full generality.


Example
-------

We create an interactive expression by calling the ``Data`` constructor on any
object or URI with which Blaze is familiar.

.. code-block:: python

   >>> from blaze import *
   >>> iris = Data('sqlite:///blaze/examples/data/iris.db::iris')  # an interactive expression
   >>> iris
       sepal_length  sepal_width  petal_length  petal_width      species
   0            5.1          3.5           1.4          0.2  Iris-setosa
   1            4.9          3.0           1.4          0.2  Iris-setosa
   2            4.7          3.2           1.3          0.2  Iris-setosa
   3            4.6          3.1           1.5          0.2  Iris-setosa
   4            5.0          3.6           1.4          0.2  Iris-setosa
   5            5.4          3.9           1.7          0.4  Iris-setosa
   6            4.6          3.4           1.4          0.3  Iris-setosa
   7            5.0          3.4           1.5          0.2  Iris-setosa
   8            4.4          2.9           1.4          0.2  Iris-setosa
   9            4.9          3.1           1.5          0.1  Iris-setosa
   ...

   >>> iris.species.<tab>  # doctest: +SKIP
   iris.species.columns       iris.species.max
   iris.species.count         iris.species.min
   iris.species.count_values  iris.species.ndim
   iris.species.distinct      iris.species.nunique
   iris.species.dshape        iris.species.relabel
   iris.species.expr          iris.species.resources
   iris.species.fields        iris.species.schema
   iris.species.head          iris.species.shape
   iris.species.isidentical   iris.species.sort
   iris.species.label         iris.species.species
   iris.species.like          iris.species.to_html
   iris.species.map

   >>> iris.species.distinct()
              species
   0      Iris-setosa
   1  Iris-versicolor
   2   Iris-virginica



In the case above ``iris`` is a ``Symbol``, just like any normal Blaze leaf
expresion

.. code-block:: python

   >>> isinstance(iris, Symbol)
   True

But ``iris`` has one additional field, ``iris.data`` which points to
a Blaze ``SQL`` object holding a SQLAlchemy Table.

.. code-block:: python

   >>> iris.data                                 # doctest: +SKIP
   <blaze.data.sql.SQL at 0x7f0f64ffbdd0>

Compute calls including ``iris`` may omit the customary namespace, e.g.

.. code-block:: python

   >>> expr = iris.species.distinct()

   >>> # compute(expr, {iris: some_sql_object})  # Usually provide a namespace
   >>> compute(expr)                             # doctest: +SKIP
   ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

This implicit namespace can be found with the ``._resources`` method

.. code-block:: python

   >>> expr._resources()                          # doctest: +SKIP
   {iris: <blaze.data.sql.SQL at 0x7f0f64ffbdd0>}

Additionally, we override the ``__repr__`` and ``_repr_html_`` methods to
include calls to ``compute``.  This way, whenever an expression is printed to
the screen a small computation is done to print the computed data instead.

As an example, this ``__repr__`` function looks something like the following:

.. code-block:: python

   def __repr__(expr):
       expr = expr.head(10)         # Only need enough to print to the screen
       result = compute(expr)       # Do the work necessary to get a result
       df = into(DataFrame, result) # Shove into a DataFrame
       return repr(df)              # Use pandas' nice printing

   Expr.__repr__ = __repr__   # Override normal __repr__ method

This provides smooth interactive feel of interactive expressions.  Work is only
done when an expression is printed to the screen and excessive results are
avoided by wrapping all computations in a ``.head(10)``.
