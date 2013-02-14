==============
 Blaze format
==============

This is a working document with ideas about blaze format.

Blaze-format is a persitence format intended to support *huge* arrays.
It provides a fast compressed format and it is sharded at the filesystem
level.

An array in blaze format appears in the filesystem as a directory. That
directory contains files that describe the data stored in the array
(the metadata) and the data itself.

The metadata as-of-now is stored as a json file. The data is partitioned
and stored in shards. The shards are stored in Bloscpack
format. Bloscpack provides time-efficient compression.

Using compression serves two purposes:
 - Reduce the size of data in the different memory layers.
 - Minimize communication costs between memory hierarchy levels
   (including disk) and thus, increase performance

The usage of time-efficient compression allows the data to be stored
compressed in-memory. Loading compressed data into the cache prior to
operation can increase performance if the decompression is fast enough.
This has been already tested with blosc.

Metadata
========
Metadata describes the schema of the stored data. It contains key
information to access the data stored. This includes information related
to the datashape of the array.

Data
====
Data is stored in shards. Each shard contains a vertical slice of the
array data stored in bloscpack format. Bloscpack organizes its data in
chunks of compressed blocks of data.

This means that data for an array is organized in 3 levels:

 - Shard. Is the filesystem unit. Data is organized in the filesystem at
   this level. Data can be moved around the filesystem at the shard
   level.

 - Chunk. Inside a Shard, data is organized in Chunks. Data is mapped
   into memory at the chunk level.

 - Block. Inside a Chunk, data is organized in Blocks. Blocks are the
   unit of compression/decompression.

You can think the shard as the storage management unit. The chunk is the
disk i/o unit and the block is the decompression unit. In order to be
maximize efficiency, if operating over a full array, the block should
act as a base unit for processing, chunks should be a base unit at the
process level. Shard is more related to storage, but if distributed
storage is used the computation should be performed in the node that has
more efficient access to the shard.
