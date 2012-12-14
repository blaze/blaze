===========
Quickstart
===========

Blaze Arrays
~~~~~~~~~~~~

::

    from blaze import Array, dshape
    ds = dshape('2, 2, int')

    a = Array([1,2,3,4], ds)


::

    >>> a
    Array
      datashape := 2 2 int64
      values    := [CArray(ptr=36111200)]
      metadata  := [manifest]
      layout    := Chunked(dim=0)

    [[1, 2],
     [3, 4]]


Disk Backed Array
~~~~~~~~~~~~~~~~~

::

    >>> import numpy as np
    >>> from blaze import Array, dshape, params

    >>> data = np.zeros(4).reshape(2,2)
    >>> Array(data, dshape('(2,2), float64'), params=params(storage='a'))
    Array
      datashape := 2, 2, float64
      values    := [CArray(ptr=56992176)]
      metadata  := [manifest, arraylike]
      layout    := Chunked(dim=0)
    [[ 0.  0.]
     [ 0.  0.]]

::

    >>> from blaze import open
    >>> open('a')
    Array
      datashape := 2, 2, float64
      values    := [CArray(ptr=56992176)]
      metadata  := [manifest, arraylike]
      layout    := Chunked(dim=0)
    [[ 0.  0.]
     [ 0.  0.]]


Custom DShapes
~~~~~~~~~~~~~~

::

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


.. Uncomment this when a way to remove the 'toplevel' from description
.. would be found...
.. Top level functions
.. ~~~~~~~~~~~~~~~~~~~

.. .. automodule:: blaze.toplevel
..    :members:
