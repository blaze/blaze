===================
Type Reconstruction
===================

The goal of type reconstruction is to resolve the relationships between
type variables and concrete variables within the context of a of an
expression but without the need for the entire expression to be explicitly
annotated.

Holding type variables as first order terms in the signatures encodes
the fact that a term can be used in many concrete contexts with
different concrete types. The end result of successful type
reconstruction and unification is to determine the least general
type satisfying the constraints on the expression.

Type inference is performed on the fly, to allow early error detection.

Signatures
----------

The signature::

    (a, b) -> c

u for a function of two arguments, the first of type ``a``, the second
of type ``b`` and returning type ``c``.

Would be this in Python 3 signature notation::

    def (x : a, y : b) -> c:
        pass

Type System
-----------

A type system in Blaze is collection of three functions::

    unifier :: (ty, ty) -> ty
    typeof :: value -> ty

And a collection of type objects with two special types::

    (?)   - dynamic type
    (top) - top type

This is specified by a namedtuple of the form::

    typesystem = namedtuple('TypeSystem', 'unifier, top, dynamic, typeof')

The type system over Blaze expression expressions is referred to
as ``BlazeT``.

Dynamic
-------

A dynamic type written as ( ``?`` ). It allows explicit down casting and
upcasting from any type to any type. In Blaze we use this to represent
opaque types that afford no specialization.

Rigid & Free
------------

Rigid type variables are those that appear in multiple times in the
signature::

     f : (a -> b -> a) -> c

     Rigid : [a]
     Free  : [b,c]

Context
-------

A context records the lexically-bound variables during the progression
of the type inference algorithm. It is a stateful dict passed through
the unifiers.

Constraints
-----------

The blaze type system records constraints over any free type variables in the
inputs. Constraints specify coercion semantics. For dimension variables it
indicates broadcastability, e.g. `(1, 10)`. For the measure (or element type)
it encodes the coercion between two types. The coercion relation is reflexive,
anti-symmetric and transitive, and forms a partial order with `top` as the
top (⊤). We only allow widening or up-casts, without loss of information.


API
~~~

.. .. automodule:: blaze.type_reconstruction
..    :members:
