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
      metadata  := Meta(MANIFEST=True, )
      layout    := Chunked(dim=0)

    [[1, 2],
     [3, 4]]


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
