API
===

This page contains a comprehensive list of functionality within ``blaze``.
Docstrings should provide sufficient understanding for any individual function
or class.


Interactive Use
---------------

.. currentmodule:: blaze.interactive

.. autosummary::

   _Data


Expressions
-----------

.. currentmodule:: blaze.expr.expressions

.. autosummary::

   Projection
   Selection
   Label
   ReLabel
   Map
   Apply
   Coerce
   Coalesce
   Cast

.. currentmodule:: blaze.expr.collections

.. autosummary::

   Sort
   Distinct
   Head
   Merge
   Join
   Concat
   IsIn

.. currentmodule:: blaze.expr.split_apply_combine

.. autosummary::

   By

Blaze Server
------------

.. currentmodule:: blaze.server.server

.. autosummary::

   Server

.. currentmodule:: blaze.server.client

.. autosummary::

   Client

Additional Server Utilities
---------------------------

.. currentmodule:: blaze.server.server

.. autosummary::

   expr_md5
   to_tree
   from_tree

.. currentmodule:: blaze.server.spider

.. autosummary::

   data_spider
   from_yaml

Definitions
-----------

.. automodule:: blaze.interactive
   :members:

.. automodule:: blaze.server.spider
   :members:

.. automodule:: blaze.server.server
   :members:

.. automodule:: blaze.server.client
   :members:

.. automodule:: blaze.expr.collections
   :members:

.. automodule:: blaze.expr.expressions
   :members:

.. automodule:: blaze.expr.reductions
   :members:

.. automodule:: blaze.expr.arrays
   :members:

.. automodule:: blaze.expr.arithmetic
   :members:

.. automodule:: blaze.expr.math
   :members:

.. automodule:: blaze.expr.broadcast
   :members:

.. automodule:: blaze.expr.datetime
   :members:

.. automodule:: blaze.expr.split_apply_combine
   :members:
