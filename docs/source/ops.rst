=========
Operators
=========

Operators are graph nodes taking homogeneously typed arguments and
returning a value prescribed by the input types. Just like NumPy
they are generally type and shape ad-hoc polymorphic.

The method by which we determine which low-level kernel to dispatch the
operator to is determined by the typechecker as a function of the
input types of the operator.

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

.. automodule:: ndtable.expr.ops
   :members:
   :undoc-members:
