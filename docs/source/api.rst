API
===

This page contains a comprehensive list of functionality within ``blaze``.
Docstrings should provide sufficient understanding for any individual function.


Usability
---------

.. currentmodule:: blaze.api.table

.. autosummary::
   Table


Table Expressions
-----------------

.. currentmodule:: blaze.expr.table

.. autosummary::
   TableSymbol
   Projection
   Column
   Selection
   ColumnWise
   Reduction
   Sort
   Distinct
   Head
   Label
   ReLabel
   Map
   Merge
   Union
   By
   Join

Data Server
-----------

.. currentmodule:: blaze.server.server

.. autosummary::
   Server
   to_tree
   from_tree

.. currentmodule:: blaze.server.client

.. autosummary::
   ExprClient



Definitions
-----------

.. automodule:: blaze.api.table
   :members:

.. automodule:: blaze.expr.table
   :members:

.. automodule:: blaze.server.server
   :members:

.. automodule:: blaze.server.client
   :members:
