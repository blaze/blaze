Split-Apply-Combine -- Grouping
===============================

Grouping operations break a table into pieces and perform some reduction on
each piece.  Consider the ``iris`` dataset:

.. code-block:: python

   >>> from blaze import data, by
   >>> from blaze.utils import example
   >>> d = data('sqlite:///%s::iris' % example('iris.db'))
   >>> d  # doctest: +SKIP
       sepal_length  sepal_width  petal_length  petal_width      species
   0            5.1          3.5           1.4          0.2  Iris-setosa
   1            4.9          3.0           1.4          0.2  Iris-setosa
   2            4.7          3.2           1.3          0.2  Iris-setosa
   3            4.6          3.1           1.5          0.2  Iris-setosa
   4            5.0          3.6           1.4          0.2  Iris-setosa


We find the average petal length, grouped by species:

.. code-block:: python

   >>> by(d.species, avg=d.petal_length.mean())
              species    avg
   0      Iris-setosa  1.462
   1  Iris-versicolor  4.260
   2   Iris-virginica  5.552

Split-apply-combine operations are a concise but powerful way to describe many
useful transformations.  They are well supported in all backends and are
generally efficient.


Arguments
---------

The ``by`` function takes one positional argument, the expression on which we
group the table, in this case ``d.species``, and any number of keyword
arguments which define reductions to perform on each group.  These must be
named and they must be reductions.

.. code-block:: python

   >>> by(grouper, name=reduction, name=reduction, ...)  # doctest: +SKIP

.. code-block:: python

   >>> by(d.species, minimum=d.petal_length.min(),
   ...               maximum=d.petal_length.max(),
   ...               ratio=d.petal_length.max() - d.petal_length.min())
              species  maximum  minimum  ratio
   0      Iris-setosa      1.9      1.0    0.9
   1  Iris-versicolor      5.1      3.0    2.1
   2   Iris-virginica      6.9      4.5    2.4


Limitations
-----------

This interface is restrictive in two ways when compared to in-memory dataframes
like ``pandas`` or ``dplyr``.

1.  You must specify both the grouper and the reduction at the same time
2.  The "apply" step must be a reduction

These restrictions make it *much* easier to translate your intent to databases
and to efficiently distribute and parallelize your computation.


Things that you can't do
------------------------

So, as an example, you can't "just group" a table separately from a reduction

.. code-block:: python

   >>> groups = by(mytable.mycolumn)  # Can't do this  # doctest: +SKIP

You also can't do non-reducing apply operations (although this could change for
some backends with work)

.. code-block:: python

   >>> groups = by(d.A, result=d.B / d.B.max())  # Can't do this  # doctest: +SKIP
