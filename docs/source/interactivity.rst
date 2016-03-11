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

We create an interactive expression by calling the ``data`` constructor on any
object or URI with which Blaze is familiar.

.. code-block:: python

   >>> from blaze import data, Symbol
   >>> from blaze.utils import example
   >>> db = data('sqlite:///%s' % example('iris.db'))  # an interactive expression
   >>> db.iris
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

   >>> db.iris.species.<tab>  # doctest: +SKIP
   db.iris.species.columns       db.iris.species.max
   db.iris.species.count         db.iris.species.min
   db.iris.species.count_values  db.iris.species.ndim
   db.iris.species.distinct      db.iris.species.nunique
   db.iris.species.dshape        db.iris.species.relabel
   db.iris.species.expr          db.iris.species.resources
   db.iris.species.fields        db.iris.species.schema
   db.iris.species.head          db.iris.species.shape
   db.iris.species.isidentical   db.iris.species.sort
   db.iris.species.label         db.iris.species.species
   db.iris.species.like          db.iris.species.to_html
   db.iris.species.map

   >>> db.iris.species.distinct()
              species
   0      Iris-setosa
   1  Iris-versicolor
   2   Iris-virginica


In the case above ``db`` is a ``Symbol``, just like any normal Blaze leaf
expresion

.. code-block:: python

   >>> isinstance(db, Symbol)
   True

But ``db`` has one additional field, ``db.data`` which points to
a SQLAlchemy Table.

.. code-block:: python

   >>> db.data                                 # doctest: +SKIP
   <sqlalchemy.Table at 0x7f0f64ffbdd0>

Compute calls including ``db`` may omit the customary namespace, e.g.

.. code-block:: python

   >>> from blaze import compute
   >>> expr = db.iris.species.distinct()

   >>> # compute(expr, {db: some_sql_object})  # Usually provide a namespace
   >>> compute(expr)                             # doctest: +SKIP
   ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

This implicit namespace can be found with the ``._resources`` method

.. code-block:: python

   >>> expr._resources()                          # doctest: +SKIP
   {db: <sqlalchemy.Table object>}

Additionally, we override the ``__repr__`` and ``_repr_html_`` methods to
include calls to ``compute``.  This way, whenever an expression is printed to
the screen a small computation is done to print the computed data instead.

As an example, this ``__repr__`` function looks something like the following:

.. code-block:: python

   from odo import odo
   from pandas import DataFrame
   from blaze import Expr

   def __repr__(expr):
       expr = expr.head(10)         # Only need enough to print to the screen
       result = compute(expr)       # Do the work necessary to get a result
       df = odo(result, DataFrame) # Shove into a DataFrame
       return repr(df)              # Use pandas' nice printing

   Expr.__repr__ = __repr__   # Override normal __repr__ method

This provides smooth interactive feel of interactive expressions.  Work is only
done when an expression is printed to the screen and excessive results are
avoided by wrapping all computations in a ``.head(10)``.
