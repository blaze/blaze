=======
Layouts
=======

Scalar layouts are arangements of scalar matrix data on to blocks
of memory that may or may not be contigious.

- Chunked
- Multichunks

If we build a composite block structure as some union of two
multiple blocks of data, then we need only have a invertible linear
transformation which is able to map coordinates between the local
coordinates of the chunks as we pull bytes from them.

::

    ChunkedL(dim=0)
    +---------+
    |         |
    |---------|
    |         |
    +---------+

::

    ChunkedL(dim=1)
    +---------+
    |    |    |
    |    |    |
    |    |    |
    +---------+


::

    Multichunk(points=...)
    +---------+
    |    |    |
    |----|----|
    |    |    |
    +---------+


Combining Chunks
~~~~~~~~~~~~~~~~

::

    f : (i,j) -> (i', j')
    g : (i',j') -> (i, j)
    f . g = id


::

      Block 1            Block 2

      1 2 3 4            1 2 3 4
    0 - - - -  vstack  0 * * * *
    1 - - - -          1 * * * *

              \     /

               1 2 3 4
             0 - - - - + --- Transform =
             1 - - - -       (i,j) -> (i, j)
             2 * * * * + --- Transform =
             3 * * * *       (i,j) -> (i-2, j)



::

       Block 1            Block 2

       1 2 3 4            1 2 3 4
     0 - - - -  hstack  0 * * * *
     1 - - - -          1 * * * *

               \     /

                1 2 3 4
              0 - - * *
              1 - - * *
              2 - - * *
              3 - - * *
                |   |
                |   |
                |   + Transform =
                |     (i,j) -> (i-2, j)
                + --- Transform =
                      (i,j) -> (i, j)


.. automodule:: ndtable.layouts.scalar
   :members:
