=========
Operators
=========

Polymorphism
~~~~~~~~~~~~

Operators are graph nodes taking homogeneously typed arguments and
returning a value prescribed by the input types. Just like NumPy
they are generally type and shape ad-hoc polymorphic.

Consider for example the ``(+)`` operator in NumPy:

.. code-block:: python

    from numpy import int32, float32, array, str_

    >>> int32(2) + int32(2)
    4

    >>> int32(2) + float32(2)
    4.0

    >>> str_('x') + str_('y')
    'xy'

    >>> array([1,2,3], dtype='int') + 1
    array([2, 3, 4])

    >>> array([1,2,3], dtype='int') + 1.0
    array([2., 3., 4.])

    >>> array([True,False,True]) + True
    array([True, True, True], dtype='bool')


Just using one operator we obtain 6 different behaviors depending
on inputs types. In Blaze each of these behaviors represents a
different function chosen at the compilaton stage informed by the
datashape of the inputs.

Properties
~~~~~~~~~~

Operators also have a variety of properties that optionally can be used
to be inform specialization and expression graph rewriting for domain
experts. ( Not all are defined for every operator. )

* ``signature``
    Map signature
* ``dom``
    Domain
* ``cod``
    Codomain
* ``nin``
    Number of elements in the domain
* ``nout``
    Number of elements in the codomain
* ``associative``
* ``commutative``
* ``leftidentity``
* ``rightidentity``
* ``identity``
* ``leftzero``
* ``rightzero``
* ``zero``
* ``leftinverse``
* ``rightinverse``
* ``inverse``
* ``distributesover``
* ``idempotent``
* ``absorptive``
* ``involutive``
* ``sideffectful``

.. automodule:: blaze.expr.ops
   :members:
