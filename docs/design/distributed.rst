Blaze Distributed
=================

This document describes the roadmap to supporting many types of different type of parallel execution.

Design Philosophy (Zen of Blaze)
--------------------------------

- Blaze describes data and computes at the data location. 
- Blaze does not schedule your compute, it only computes.
- Blaze does not enforce a store, it works with how you store your data.
- Blaze is lazy.

Use Cases
+++++++++

A.  A single array that cannot fit into memory of a node.
  - Compute aggregate functions for basic analysis
  - Pass data to scalable numerics package for advanced analysis (almost all are MPI)
B.  A multiple arrays each on a different process.
  - Data streams from different APIs

Basic Design iteration 1
------------------------

- DistArray class that acts a the top level interactive interface.
- Build a distributed data descriptor:
  - Data descriptor requires a Communicator, Serializer, and Data Decomposition
  - First use a ZMQ data descriptor, Pickle Serialization, Block decomposition
  - Second use an MPI data descriptor, MPI Serialization, Block decomposition
  - Third use a HDFS data descriptor, Pickle Serialization, Unknown decomposition

Distributed Data Descriptor
+++++++++++++++++++++++++++

- Give a variety of choices on Communication, Serialization, and Decomposition.
- All combinations of choices will not be suitable, but where reasonable they should be implemented
- Performance will heavily depend on choosing the right algorithms given the communication and decomposition patterns, Blaze does not get in the way, it only wants to analyze your data
- Should add get_block and set_block to allow communicator to hide network latency.

Communicator
++++++++++++

Basic remote procedure calling mechanisms:
- Send blaze functions if serializer allows, otherwise uses canned set of opcodes.
- Determines allowed communication patterns (Pub-sub, collective comm, map-reduce)

Serializer
++++++++++

- Use Pickle (or Dill) where possible
- MPI causes much pain if one tries to use anything but MPI types. Needs a simple dumb serialization.

Decomposition
+++++++++++++

For a first pass only split outermost dimension.

- Block decomposition (Given block_size (arr_len/num_nodes), node p owns array[block_size * p: block_size * (p+1)]
  - Very easy to address
  - Not as dynamic as one likes
- Variadic block decomposition (maintain global mapping of machine sizes)
- Unknown decomposition
  - Cannot assume sizes and must query to find data
  - Fine for map-reduce but terrible for off node-access

Extensions to DataShape
-----------------------

- Use case A needs concept of local size versus global size. 
- Needs location semantics
  - list of uri’s and decomposition
