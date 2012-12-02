=====
Graph
=====

All nodes are one of the following types:

Node Types
~~~~~~~~~~

APP
    Application of a function or operator to a set of values.

OP
    Operator a mapping between multiple operands returning a
    single result.

VAL
    A value providing bytes.

Node Instances
~~~~~~~~~~~~~~

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

.. ditaa::

                +---+
           Dom -| f |-> Cod
                +---+

.. ditaa::

              +---------+
    a  ----+  |         |
           |  |   Op    |
    b  ----+--|---------|--> d
           |  |         |
    c  ----+  |         |
              +---------+

.. ditaa::

    Dom ---------------------> Cod

             +------------+
             |    App     |
    ty --+   |  +------+  |
         |   |  |      |  |
    ty --+---|--| Op   |--|--> d
         |   |  |      |  |
    ty --+   |  +------+  |
             +------------+

.. automodule:: ndtable.expr.graph
   :members:
