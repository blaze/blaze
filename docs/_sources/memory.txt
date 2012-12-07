=============
Memory Layout
=============

A chunked array behaves similar to a Numpy array but is composed
of multiple noncontiguous buffers in memory.

The chunks are bits of data compressed as a whole, but that can be
decompressed partially in order to improve the fetching of small parts
of the array. This chunked nature of the carray objects, together with
a buffered I/O, makes appends very cheap and fetches reasonably fast
(although the modification of values can be an expensive operation).

The compression/decompression process is carried out internally by
Blosc, a high-performance compressor that is optimized for binary data.
That ensures maximum performance for I/O operation.

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

                             nbytes          blocksize          ctbytes

      0 | 1 | 2 | 3      4 | 5 | 6 | 7     8 | 9 | A | B     C | D | E | F
      ^   ^   ^   ^
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------versionlz
      +--------------version

