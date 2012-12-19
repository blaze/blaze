=========================================================================
Proposal for a persistent format that can include variable length objects
=========================================================================

:Author: Francesc Alted

This document proposes a new version of the Bloscpack format ([1]_)
that accommodates variable length data.  For simplicity, I will be focusing
on the monolithic flavor of the original proposal.  A proposal for the
chunked flavor should follow as soon as we reach an agreement.

.. [1] https://github.com/esc/bloscpack


Monolithic format
=================

Header format
-------------

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

    :``bit 2 (0x04)``:
        If this dataset is variable length.

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

The overall data-size can be computed as ``chunk-size * (nchunks - 1) +
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

  {'dtype': 'varchar', 'shape': [1024], 'others': []}

Description of the offsets entries for data items
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Offsets of the variable length items into the file are used for
stablishing the boundaries between items, and these are mandatory for
variable length data.

Each offset is a 64 bit signed little-endian integer (``int64``). A
value of ``-1`` denotes an unknown offset.  Initially, all offsets
should be initialized to ``-1`` and filled in after writing all
items. Thus, if the compression of the file fails prematurely or is
aborted, all offsets should have the value ``-1``.  Each offset
denotes the exact position of the item in the file such that seeking
to the offset, will position the file pointer at the start of the
desired chunk. The layout of this section is then::

    |-previous-section-|-offset-|-offset-|...|-chunk-|-chunk-|...|


Description of the chunk format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A chunk will be keeping one and only one variable length object.

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

As there will be a chunk for every variable length data object, that
means that it could be in the order of 40 ~ 100 bytes for every data
entry, and that could be a lot for small objects (i.e. one alternate
system must be found for this scenario).
