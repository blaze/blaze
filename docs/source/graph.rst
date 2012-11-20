=====
Graph
=====

All nodes are one of the following types::

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

The core graph node types in Blaze are 

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


.. ditaa::

         +---+
    Dom -| f |-> Cod 
         +---+

.. ditaa::

    Dom                             Cod
              +----------------+
              |       +----+   |
    a  ----+  |       |    |   |
           |  |       |    |   |
    b  ----+--|-------| Op |---|--> d
           |  |       |    |   |
    c  ----+  |       |    |   |
              |       +----+   |
              +----------------+


.. automodule:: ndtable.expr.graph
   :members:
