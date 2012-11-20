=============
Memory Layout
=============

A chunked array behaves similar to a Numpy array but is composed
of multiple discontigious buffers in memory.

.. ditaa::

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

.. ditaa::

                +--------------+
    Header      |              | 16 bytes
                +--------------+
    Data        |              | ctbytes
                +--------------+


The first 16 bytes in the buffer are the header of the chunk. The
first four are simply bytes, the last three are are each unsigned ints
(uint32) each occupying 4 bytes. The header is always little-endian.
'ctbytes' is the length of the buffer including header and nbytes is the
length of the data when uncompressed.

.. ditaa::


      0 | 1 | 2 | 3      4 | 5 | 6 | 7     8 | 9 | A | B     C | D | E | F
      ^   ^   ^   ^          nbytes          blocksize          ctbytes
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------versionlz
      +--------------version

