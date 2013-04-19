==============
 Blaze format
==============

This is a working document with ideas about blaze format.

Blaze-format is a persistence format intended to support *huge* arrays.
It provides a fast compressed format that is also partitioned at the
filesystem level.

An array in blaze format appears in the filesystem as a directory. That
directory contains files that describe the data stored in the array
(the metadata) and the data itself.

The metadata as-of-now is stored as a JSON file. The data is partitioned
and stored in shards. The shards are stored in Bloscpack
format. Bloscpack provides time-efficient compression.

Using compression serves two purposes:
 - Reduce the size of data in the different memory layers.
 - Minimize communication costs between memory hierarchy levels
   (including disk) and thus, increase performance

The usage of time-efficient compression allows the data to be stored
compressed in-memory. Loading compressed data into the cache prior to
operation can increase performance if the decompression is fast enough.
This has been already tested with Blosc.

Metadata
========
Metadata describes the schema of the stored data. It contains key
information to access the data stored. This includes information related
to the datashape of the array.

Data
====
Data is stored in super-chunks. Each super-chunk contains a horizontal slice of the
array data stored in Bloscpack format. Bloscpack organizes its data in
chunks of compressed blocks of data.

This means that data for an array is organized in 3 levels:

 - Super-chunk. Is the filesystem unit. Data is organized in the filesystem at
   this level. Data can be moved around the filesystem at the super-chunk
   level. A Super-chunk can provide a natural unit for sharding.

 - Chunk. Inside a super-chunk, data is organized in Chunks. Data is mapped
   into memory at the chunk level.

 - Block. Inside a Chunk, data is organized in Blocks. Blocks are the
   basic unit of compression/decompression.

You can think the super-chunk as the storage management unit. The chunk
is the disk i/o unit and the block is the decompression unit. In order
to be maximize efficiency, if operating over a full array, the block
should act as a base unit for processing, chunks should be a base unit
at the process level. A super-chunk is more related to storage, but if
distributed storage is used the computation should be performed in the
node that has more efficient access to that super-chunk (providing
sharding).

===================
 Related solutions
===================

SciDB
=====
SciDB is an array database. SciDB tackles objectives beyond the scope of
our file-format, as a single database may contain several arrays. SciDB
also provides and Array Query Language (AQL) and an Array Functional
Language (AFL). Those languages provide a functionality that in the our
case will be provided in other parts of Blaze.

From an array storage point of view, SciDB has the following features:

1. **Dimensions** and **Attributes**. Dimensions define the *grid* of the array,
while Attributes define the contents of a *cell* in the array. This is
somewhat less flexible than blaze capabilities. 
2. Support for non-integer dimensions. 
3. Explicit control on **Chunking**. A *chunk* size can be explicitly
especified. In fact, chunk size is specified by dimension. An overlap
can be specified.
4. Storage uses vertical partitioning (physically placing attributes
together). Run-length encoding is used to compress repeated values and a
cache of decompressed chunks is held in RAM.
5. Implements versioning. "no overwrite" storage model.


============
 References
============
scidb-userguide-13.1
