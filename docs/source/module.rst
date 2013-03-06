=============
Module System
=============

The Blaze module is a domain specific language for defining module
groups of definitions of data types and associated operations over
associated types and enforce a consistent naming scheme for these
definitions.

The Blaze module system guides for the construction composition of
proxy objects in a way that preserves the types of the operands without
resorting to immediate evaluation.

Traits
------

Trait Properties
----------------

:fun:

:op:

:ty:

Typesets
--------

Typesets are domains of types over which entire sets of traits can be
defined.

+-----------+----------------------------------------------------+
| bools     |  bool                                              |
|-----------+----------------------------------------------------|
| ints      |  int8, int16, int32, int64                         |
|-----------+----------------------------------------------------|
| uints     |  uint8, uint16, uint32, uint64                     |
|-----------+----------------------------------------------------|
| floats    |  float32, float64                                  |
|-----------+----------------------------------------------------|
| complexes |  complex64, complex128                             |
|-----------+----------------------------------------------------|
| string    |  string                                            |
|-----------+----------------------------------------------------|
| discrete  |  ints , uints                                      |
|-----------+----------------------------------------------------|
| reals     |  ints , floats                                     |
|-----------+----------------------------------------------------|
| continuous|  floats , complexes                                |
|-----------+----------------------------------------------------|
| numeric   |  discrete , continuous                             |
|-----------+----------------------------------------------------|
| temporal  |  datetime , timedelta                              |
+-----------+----------------------------------------------------+

Proxies
-------

.. code-block:: python

    # ExampleModule.py
    # implemention of ExampleModule

    def foo(At):
        return At[0]

    def bar(a):
        return [a]

    __types__ = {
        'List': (list, None),
    }

    __functions__ = {
        'foo': foo,
        'bar': bar,
    }


.. code-block:: ocaml

    module ExampleModule {

        trait Example[T a]:
            fun f0 :: a -> T a
            fun f1 :: T a -> a

        impl Example[List a]:
            fun f0 = foo
            fun f1 = bar

    }

The crux of the proxy system is the ``blaze.module.proxy`` class which
is created dynamically for each proxy object. The ``_create_proxy``
method is called each time a proxy is created and populates the Python
class' namespace with the types specified by the implementations over
it in the module definitions. Each of these are themselves wrapped in
function wrappers that return proxy objects corresponding to the output
types of the function in question. 

In this way the proxy object system preserves welltypedness of
composition of proxy objects and rejects invalid compositions ( i.e. map
over scalars ) while still preserving the ability to express NumPy like
broadcasting and implicit ceorcion logic.

.. code-block:: python

    class Proxy(object):
        __slots__ = ["_ty", "_ns", "_ctx", "_node"]

        ...

        @classmethod
        def _create_proxy(cls, ty, ctx, ns):
            namespace = {}

            for name, sig in ns:
                meth = build_bound_meth(ctx, name, sig)
                namespace[name] = meth

            namespace['capabilities'] = namespace.keys()

        ...

.. code-block:: python

    >>> x = List([1,2,3])
    >>> print x
    <Proxy List int>
    >>> f1(x)
    <Proxy int>
    >>> y = f0(x)
    >>> print y
    <Proxy int>
    >>> y.eval()
    1
