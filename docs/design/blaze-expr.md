Blaze Expressions
=================

Blaze expressions are intended to be a very high level operations on Blaze
object. This allows for the operations to be built up in a natural fashion
and optimizations be applied before the expression graph is lowered to
another intermediate representation (IR). Additionally this high level graph
allows for many parallel primitives that will help select an IR that is
amenable to the requirements

The expression graph consists of nodes, visitors, and transformers. All of which
can be registered by plugins which is useful for specialized datasources, e.g.
scidb, bayesDB, etc.

An example of using an expression graph with a blaze object is map
such a function is a composition of maps. For example a map adding one
followed by a map adding two will be transformed into a map adding three

Nodes
-----

Each node inherits from the BlazeExprNode base class and has a .args field.

Visitors
--------

Vistors are intended to traverse a Blaze Expression and call a function based on the node name.

Transformers
------------

