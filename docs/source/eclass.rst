================
Evaluation Class
================

All objects in Blaze expressions are tagged with an evaluation
class. The two evaluation classes are

:Manifest:
    Evaluation is done immedietely for every operation. Expression
    graph has at most a depth of 1.

:Delayed:
    Evaluation is delayed until explictly compiled. Expression graph has
    unlimited depth.

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

.. automodule:: blaze.eclass
   :members:
