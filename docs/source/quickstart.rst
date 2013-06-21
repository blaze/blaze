===========
Quickstart
===========

Blaze Arrays
~~~~~~~~~~~~

::

    >>> from blaze import array, dshape
    >>> ds = dshape('2, 2, int32')
    >>> a = array([[1,2],[3,4]], ds)

::

    >>> a
    array([[1, 2],
           [3, 4]],
          dshape='2, 2, int32')


Disk Backed Array
~~~~~~~~~~~~~~~~~

::

    >>> from blaze import array, dshape, Persist
    >>> p = Persist('foo.blz')
    >>> ds = dshape('2, 2, int32')
    >>> a = array([[1,2],[3,4]], ds, persist=p)


::

    >>> a
    array([[1, 2],
           [3, 4]],
           dshape='2, 2, int32')

::

    >>> from blaze import open, drop, Persist
    >>> open(Persist('foo.blz'))
    array([[1, 2],
           [3, 4]],
           dshape='2, 2, int32')

    >>> drop(Persist('foo.blz'))
    

Iterators
~~~~~~~~~

::

    from blaze import fromiter, params
    a = fromiter(xrange(10), 'x, float64', params=params(clevel=5))


Custom DShapes
~~~~~~~~~~~~~~

::

    # FIXME: delete, or tweak?
    from blaze import Table, RecordDecl, derived
    from blaze import int32, string

    class CustomStock(RecordDecl):
        name   = string
        max    = int32
        min    = int32

        @derived
        def mid(self):
            return (self.min + self.max)/2


::

    >>> CustomStock
    {name:string, max: int32, min: int32, mid: int32}

    >>> a = Table([('GOOG', 120, 153)], CustomStock)


.. XXX: Added a dedicated toplevel page

.. Uncomment this when a way to remove the 'toplevel' from description
.. would be found...
.. Top level functions
.. ~~~~~~~~~~~~~~~~~~~

.. .. automodule:: blaze.toplevel
..    :members:
