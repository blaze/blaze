=============
Memory Layout
=============

A chunked array behaves similar to a Numpy array but is composed
of multiple discontigious buffers in memory.

::

                +--------------+
    Chunk 1     |              | chunksize
                +--------------+
    Chunk 2     |              | chunksize
                +--------------+
    Chunk 3     |              | chunksize
                +--------------+
    Chunk 4     |              | chunksize
                +--------------+

                +--------------+
    Leftovers   |              | variable
                +--------------+
