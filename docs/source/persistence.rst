================================
Persistent internal format (BLZ)
================================

Introduction
============

Blaze is designed to work with data that is both in memory and disk in
a transparent way.  BLZ is the format that is implemented internally
so as to persist data on-disk (although it supports in-memory starge
too).  The goals of the BLZ format are:

1. Allow to work with data directly on disk, exactly on the same way
   than data in memory.

2. The persistence layer should support the same access capabilities
   than Blaze objects including: modifying, appending and removing data,
   as well as direct access to data (in the same way than RAM).

3. Transparent data compression must be possible.

4. User metadata addition must be possible too.

5. Data objects are allowed to be enlarged or shrunk.

6. Data is not allowed to be modified.

7. And last but not least, the data should be easily 'shardable' for
   optimal behavior in distributed storage.  Providing a format that is
   already 'sharded' by default would represent a big advantage for
   allowing spreading a Blaze object among different nodes.

These points, in combination with a distributed filesystem, and with a
system that would be aware of the physical topology of the underlying
infrastructure would allow to largely avoid the need for a
Disco/Hadoop infrastructure, permitting much better flexibility and
performance.

The data files will be made of a series of chunks put together using
the Blosc metacompressor by default.  Blosc being a metacompressor,
means that it can use different compressors and filters, while
leveraging its blocking and multithreading capabilities.

The low level description for the BLZ format follows.  It must be
noted that with Blaze 0.1 the implementation is almost complete,
except for the fact that that superchunks are not yet there.


The BLZ format
==============

The layout
----------

For every dataset, it will be created a directory, with a
user-provided name that, for generality, we will call it `root` here.
The root will have another couple of subdirectories, named data and
meta::

        root  (the name of the dataset)
        /  \
     data  meta

The `data` directory will contain the actual data of the dataset,
while the `meta` will contain the metainformation (dtype, shape,
chunkshape, compression level, filters...).

The `data` layout
-----------------

Data will be stored by what is called a `superchunk`, and each
superchunk will use exactly one file.  The size of each superchunk
will be decided automatically by default, but it could be specified by
the user too.

Each of these directories will contain one or more superchunks for
storing the actual data.  Every data superchunk will be named after
its sequential number.  For example::

    $ ls data
    __1__.bin  __2__.bin  __3__.bin  __4__.bin ... __1030__.bin

This structure of separate superchunk files allows for two things:

1. Datasets can be enlarged and shrink very easily
2. Horizontal sharding in a distributed system is possible (and cheap!)

At its time, the `data` directory might contain other subdirectories
that are meant for storing components for a 'nested' dtype (i.e. an
structured array, stored in column-wise order)::

        data  (the root for a nested datatype)
        /  \     \
     col1  col2  col3
          /  \
        sc1  sc3

This structure allows for quick access to specific chunks of columns
without a need to load the complete dataset in memory.


The `superchunk` layout
-----------------------

Here it is how the superchunks are going to be laid out.  It is worth
to mention that this format will be based on the Bloscpack format [1]_
and that it will continue to evolve the next future.

.. [1] https://github.com/esc/bloscpack

Header format
~~~~~~~~~~~~~

The design goals of the header format are to contain as much
information as possible to achieve interesting things in the future
and to be as general as possible so as to ensure compatibility with
the chunked persistence format.

The following ASCII representation shows the layout of the header::

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
    | b   l   p   k | ^ | ^ | ^ | ^ |   chunk-size  |  last-chunk   |
                      |   |   |   |
          version ----+   |   |   |
          options --------+   |   |
         checksum ------------+   |
         typesize ----------------+

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
    |            nchunks            |   meta-size   |   RESERVED    |


The first 4 bytes are the magic string ``blpk``. Then there are 4
bytes, the first three are described below and the last one is
reserved. This is followed by 4 bytes for the ``chunk-size``, another
4 bytes for the ``last-chunk-size`` and 8 bytes for the number of
chunks. Finally, the ``meta-size`` accounts for the amount of bytes
that takes the metadata to be stored.  The last 4 bytes are reserved
for use in future versions of the format.

Effectively, storing the number of chunks as a signed 8 byte integer,
limits the number of chunks to ``2**63-1 = 9223372036854775807``, but
this should not be relevant in practice, since, even with the moderate
default value of ``1MB`` for chunk-size, we can still stores files as
large as ``8ZB`` (!) Given that in 2012 the maximum size of a single
file in the Zettabye File System (zfs) is ``16EB``, Bloscpack should
be safe for a few more years.

Description of the header entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All entries are little-endian.

:version:
    (``uint8``)
    format version of the Bloscpack header, to ensure exceptions in case of
    forward incompatibilities.
:options:
    (``bitfield``)
    A bitfield which allows for setting certain options in this file.

    :``bit 0 (0x01)``:
        If the offsets to the chunks are present in this file.

    :``bit 1 (0x02)``:
        If metadata is present in this file.

:checksum:
    (``uint8``)
    The checksum used. The following checksums, available in the python
    standard library should be supported. The checksum is always computed on
    the compressed data and placed after the chunk.

    :``0``:
        ``no checksum``
    :``1``:
        ``zlib.adler32``
    :``2``:
        ``zlib.crc32``
    :``3``:
        ``hashlib.md5``
    :``4``:
        ``hashlib.sha1``
    :``5``:
        ``hashlib.sha224``
    :``6``:
        ``hashlib.sha256``
    :``7``:
        ``hashlib.sha384``
    :``8``:
        ``hashlib.sha512``
:typesize:
    (``uint8``)
    The typesize of the data in the chunks. Currently, assume that the typesize
    is uniform. The space allocated is the same as in the Blosc header.
:chunk-size:
    (``int32``)
    Denotes the chunk-size. Since the maximum buffer size of Blosc is 2GB
    having a signed 32 bit int is enough (``2GB = 2**31 bytes``). The special
    value of ``-1`` denotes that the chunk-size is unknown or possibly
    non-uniform.
:last-chunk:
    (``int32``)
    Denotes the size of the last chunk. As with the ``chunk-size`` an ``int32``
    is enough. Again, ``-1`` denotes that this value is unknown.
:nchunks:
    (``int64``)
    The total number of chunks used in the file. Given a chunk-size of one
    byte, the total number of chunks is ``2**63``. This amounts to a maximum
    file-size of 8EB (``8EB = 2*63 bytes``) which should be enough for the next
    couple of years. Again, ``-1`` denotes that the number of is unknown.

The overall file-size can be computed as ``chunk-size * (nchunks - 1) +
last-chunk-size``. In a streaming scenario ``-1`` can be used as a placeholder.
For example if the total number of chunks, or the size of the last chunk is not
known at the time the header is created.

Description of the metadata section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section goes after the header, and it is just a JSON serialized
version of the metadata that is to be saved.  As JSON has its
limitations as any other serializer, only a subset of Python
structures can be stored, so probably some additional object handling
must be done prior to serialize some metadata.

Example of metadata stored:

  {'dtype': 'float64', 'shape': [1024], 'others': []}

Description of the offsets entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Offsets of the chunks into the file are to be used for accelerated
seeking. The offsets (if activated) follow the metadata section . Each
offset is a 64 bit signed little-endian integer (``int64``). A value
of ``-1`` denotes an unknown offset.  Initially, all offsets should be
initialized to ``-1`` and filled in after writing all chunks. Thus, If
the compression of the file fails prematurely or is aborted, all
offsets should have the value ``-1``.  Each offset denotes the exact
position of the chunk in the file such that seeking to the offset,
will position the file pointer such that, reading the next 16 bytes
gives the Blosc header, which is at the start of the desired
chunk. The layout of the file is then::

    |-bloscpack-header-|-offset-|-offset-|...|-chunk-|-chunk-|...|

Description of the chunk format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The header for the Blosc chunk has this format (Blosc 1.0 on)::

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
      ^   ^   ^   ^ |     nbytes    |   blocksize   |    ctbytes    |
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------blosclz version
      +--------------blosc version

Following the header there will come the compressed data itself.
Blosc ensures that the compressed buffer will not take more space than
the original one + 16 bytes (the length of the header).

At the end of each blosc chunk some empty space could be added (this
can be parametrized) in order to allow the modification of some data
elements inside each block.  The reason for the additional space is
that, as these chunks will be typically compressed, when modifying
some element of the chunk it is not guaranteed that the resulting
chunk will fit in the same space than the old one.  Having this
provision of a small empty space at the end of each chunk will allow
for storing the modified chunks in many cases, without a need to save
the entire file on a different part of the disk.

Overhead
~~~~~~~~

Depending on which configuration for the file is used a constant, or
linear overhead may be added to the file. The Bloscpack header adds 32
bytes in any case. If the data is non-compressible, Blosc will add 16
bytes of header to each chunk. If used, both the checksum and the
offsets will add overhead to the file. The offsets add 8 bytes per
chunk and the checksum adds a fixed constant value which depends on
the checksum to each chunk. For example, 32 bytes for the ``adler32``
checksum.

Also, depending on the number of reserved bytes at the end of each
chunk (the default is to not reserve them), that will add another
overhead to the final size. 


The `meta` files
----------------

Here there can be as many files as necessary.  The format for every
file will be JSON, so caution should be used for ensuring that all the
metadata can be serialized and deserialized in this format.  There
could be three (or more, in the future) files:

The `sizes` file
~~~~~~~~~~~~~~~~

This contains the shape of the dataset, as well as the uncompressed
size (``nbytes``) and the compressed size (``cbytes``).  For example::

    $ cat meta/sizes
    {"shape": [10000000], "nbytes": 80000000, "cbytes": 17316745}

The `storage` file
~~~~~~~~~~~~~~~~~~

Here comes the information about the data type, defaults and how data
is being stored.  Example::

    $ cat myarray/meta/storage
    {"dtype": "float64", "cparams": {"shuffle": true, "clevel": 5},
     "chunklen": 16384, "dflt": 0.0, "expectedlen": 10000000}

The `attributes` file
~~~~~~~~~~~~~~~~~~~~~

In this file it comes additional user information.  Example::

    $ cat myarray/meta/attributes
    {"temperature": 11.4, "scale": "Celsius",
     "coords": {"lat": 40.1, "lon": 0.5}}
