
=======================
Interactive Expressions
=======================

Internally Blaze is very abstract; this limits interactivity.  To solve this
Blaze provides *interactive expressions* which give a smooth real-time
experience to handling foreign data.

Expressions with Data
---------------------

Internally Blaze separates the intent of the computation from the data/backend.
While powerful, this abstract separation limits *interactivity*, one of the
core goals of Blaze.

To resolve this conflict Blaze provides *interactive expressions* which are
exactly like normal expressions but their leaves may hold on to a concrete data
resource (like a DataFrame or SQL database.)


Example
-------

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
expresion, but ``iris`` has one additional field, ``iris.data`` which points to
a Blaze ``SQL`` object holding a SQLAlchemy Table.

.. code-block:: python

   >>> isinstance(iris, Symbol)
   True
   >>> iris.data                                 # doctest: +SKIP
   <blaze.data.sql.SQL at 0x7f0f64ffbdd0>


Compute calls including ``iris`` may omit the customary namespace, e.g.

.. code-block:: python

   >>> expr = iris.species.distinct()

   >>> # compute(expr, {iris: some_sql_object})  # Usually provide a namespace
   >>> compute(expr)                             # Now namespace is implicit
   [u'Iris-setosa', u'Iris-versicolor', u'Iris-virginica']

This implicit namespace can be found with the ``.resources`` method

.. code-block:: python

   >>> expr.resources()                          # doctest: +SKIP
   {iris: <blaze.data.sql.SQL at 0x7f0f64ffbdd0>}

Additionally, we override the ``__repr__`` and ``_repr_html_`` methods to
include calls to ``compute``, something like the following:

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
