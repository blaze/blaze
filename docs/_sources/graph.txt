=====
Graph
=====

All nodes are one of the following types:

Kinds
~~~~~

:APP:

    Application of a function or operator to a set of values.

:OP:

    Operator a mapping between multiple operands returning a
    single result.

:FUN:

    A function mapping multiple operands to results or effects.

:VAL:

    A value providing a byte interface.

Instances
~~~~~~~~~

The core graph node types in Blaze are:

* App
* ArrayNode
* ExpressionNode
* FloatNode
* Fun
* FunApp
* IndexNode
* IntNode
* Literal
* Op
* Slice
* StringNode

The domain and codomain of an operator are determined only in the
context of an application binding.

The domain and codomain of an function are fixed by definition.

Visualization
~~~~~~~~~~~~~

TODO

API
---

.. .. autoclass:: blaze.expr.graph.App

.. .. autoclass:: blaze.expr.graph.Op

.. .. autoclass:: blaze.expr.graph.Literal
