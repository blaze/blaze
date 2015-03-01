API
===

This page contains a comprehensive list of functionality within ``blaze``.
Docstrings should provide sufficient understanding for any individual function.


Interactive Use
---------------

.. currentmodule:: blaze.interactive

.. autosummary::
   Data


Table Expressions
-----------------

.. currentmodule:: blaze.expr.table

.. autosummary::
   TableSymbol


.. currentmodule:: blaze.expr.expressions

.. autosummary::
   Projection
   Field
   Selection
   ElemWise
   Label
   ReLabel
   Map

.. currentmodule:: blaze.expr.reductions

.. autosummary::
   Reduction

.. currentmodule:: blaze.expr.collections

.. autosummary::
   Sort
   Distinct
   Head
   Merge
   Join

.. currentmodule:: blaze.expr.split_apply_combine

.. autosummary::
   By

Data Server
-----------

.. currentmodule:: blaze.server.server

.. autosummary::
   Server
   to_tree
   from_tree

.. currentmodule:: blaze.server.client

.. autosummary::
   Client



Definitions
-----------

.. automodule:: blaze.interactive
   :members:

.. automodule:: blaze.expr.table
   :members:

.. automodule:: blaze.server.server
   :members:

.. automodule:: blaze.server.client
   :members:
