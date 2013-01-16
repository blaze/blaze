================
Evaluation Class
================

All objects in Blaze expressions are tagged with an evaluation
class. The two evaluation classes are

:Manifest:
    Evaluation is done immedietely for every operation. Expression
    graph has at most a depth of 1. Shape and type checking is done at
    immedietely.

:Delayed:
    Evaluation is delayed until explictly compiled. Expression graph has
    unlimited depth. Shape and type checking is done at compile-time.

The eclass of the application of an operator to operands is
decided by mapping the following decision procedure over all
combinations of arguments.::

    def decide_eclass(a,b):
        if (a,b) == (MANIFEST, MANIFEST):
            return MANIFEST
        if (a,b) == (MANIFEST, DELAYED):
            return MANIFEST
        if (a,b) == (DELAYED, MANIFEST):
            return MANIFEST
        if (a,b) == (DELAYED, DELAYED):
            return DELAYED

In short delayed experssion are *algebraicly closed*.

.. automodule:: blaze.eclass
   :members:
