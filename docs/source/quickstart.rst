===========
Quickstart
===========

Blaze Arrays
~~~~~~~~~~~~

::

    from ndtable import Array, dshape
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

    from ndtable import Table, RecordClass, int32, string

    class Stock(RecordClass):
        name   = string
        max    = int32
        min    = int32

        def mid(self):
            return (self.min + self.max)/2

    ds = dshape(Stock)

::

    >>> ds
    {name:string, max: int32, min: int32, mid: int32}
