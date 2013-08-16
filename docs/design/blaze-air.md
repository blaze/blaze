Blaze AIR (Array Intermediate Representation)
=============================================

 * [Blaze Execution System](blaze-execution.md)

Blaze needs some kind of array IR to represent deferred computations,
which can get processed by an execution planner for out of core
and distributed computations. Some of the goals of this system
include:

 * Transport blaze expressions across the network, aka "moving code to data".
 * Represent blaze's array programming abstractions explicitly. This means
   to convert implicit broadcasting to explicit, picking particular type
   signatures of kernels within blaze functions, etc.
 * Be understandable by third party systems, and participate in the
   development/discussion (https://groups.google.com/forum/#!topic/numfocus/tX7fRfwiFkI).

TODO: Develop an initial draft of blaze AIR.