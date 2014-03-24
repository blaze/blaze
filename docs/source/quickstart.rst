Quickstart
===========

This quickstart is here to show some simple ways to get started created
and manipulating Blaze arrays. To run these examples, import blaze
as follows.

.. doctest::

    >>> import blaze
    >>> from blaze import array
    >>> from datashape import dshape

Blaze Arrays
~~~~~~~~~~~~

To create simple Blaze arrays, you can construct them from
nested lists. Blaze will deduce the dimensionality and
data type to use.

.. doctest::

    >>> array(3.14)
    array(3.14,
          dshape='float64')

.. doctest::

    >>> array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]],
          dshape='2 * 2 * int32')

You can override the data type by providing the `dshape`
parameter.

.. doctest::

    >>> array([[1, 2], [3, 4]], dshape='float64')
    array([[ 1.,  2.],
           [ 3.,  4.]],
          dshape='2 * 2 * float64')

Blaze has a slightly more general data model than NumPy,
for example it supports variable-sized arrays.

.. doctest::

    >>> array([[1], [2, 3, 4], [5, 6]])
    array([[     1],
           [     2,      3,      4],
           [     5,      6]],
          dshape='3 * var * int32')

Its support for strings includes variable-sized strings
as well.

.. doctest::

    >>> array([['test', 'one', 'two', 'three'], ['a', 'braca', 'dabra']])
    array([['test', 'one', 'two', 'three'],
           ['a', 'braca', 'dabra']],
          dshape='2 * var * string')

Simple Calculations
~~~~~~~~~~~~~~~~~~~

Blaze supports ufuncs and arithmetic similarly to NumPy.

.. doctests::
    >>> a = array([1, 2, 3])
    >>> blaze.sin(a) + 1
    array([ 1.84147098,  1.90929743,  1.14112001],
          dshape='3 * float64')
    >>> blaze.sum(3 * a)
    array(18,
          dshape='int32')

Iterators
~~~~~~~~~

Unlike in NumPy, Blaze can construct arrays directly from iterators,
automatically deducing the dimensions and type just like it does
for lists.

.. doctest::

    >>> from blaze import array, dshape
    >>> alst = [1, 2, 3]
    >>> array(alst.__iter__())
    array([1, 2, 3],
          dshape='3 * int32')

.. doctest::

    >>> array([j-i for j in range(1,4)] for i in range(1,4))
    array([[ 0,  1,  2],
           [-1,  0,  1],
           [-2, -1,  0]],
          dshape='3 * 3 * int32')

.. doctest::

    >>> from random import randrange
    >>> array((randrange(10) for i in range(randrange(5))) for j in range(4))
    array([[           7,            9],
           [           5,            2,            6,            4],
           [           9,            2,            2,            5],
           [           5]],
          dshape='4 * var * int32')


Disk Backed Array
~~~~~~~~~~~~~~~~~

Blaze uses the BLZ format for storing compressed, chunked
arrays on disk. These can be used through the Storage
object.

.. doctest::

    >>> import blaze
    >>> p = blaze.Storage('foo.blz')
    >>> ds = blaze.dshape('2 * 2 * int32')
    >>> a = blaze.array([[1,2],[3,4]], ds, storage=p)


.. doctest::

    >>> a
    array([[1, 2],
           [3, 4]],
          dshape='2 * 2 * int32')


.. doctest::

    >>> import blaze
    >>> blaze.from_blz(blaze.Storage('foo.blz'))
    array([[1, 2],
           [3, 4]],
          dshape='2 * 2 * int32')
    >>> blaze.drop(blaze.Storage('foo.blz'))




.. XXX: Added a dedicated toplevel page

.. Uncomment this when a way to remove the 'toplevel' from description
.. would be found...
.. Top level functions
.. ~~~~~~~~~~~~~~~~~~~

.. .. automodule blaze.toplevel
..    :members:
