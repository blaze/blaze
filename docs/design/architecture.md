# Blaze High Level Architecture

This design document covers the high level
architecture we would like for Blaze. It is
not an overview of the current implementation,
but rather a sketch of how the pieces should
fit together.

## Major Components

These three components form the top level
namespace of blaze.

```
blaze.core
blaze.interface
blaze.plugins
```

Two additional namespaces provide additional
utilities and acceptance tests.

```
blaze.util
blaze.tests
```

### Core

The Blaze Core defines internal interfaces,
the basic data structures, and a plugin registration
mechanism.

```
       Data Descriptor
       /      |      \
      /       |       \
     IO    Compute   Distributed
```

### Interface

This is the publicly exposed interface of Blaze,
built out of the services provided by Core.

* Array, Table objects
* UFuncs, other Blaze Functions
* Array/Table-level IO APIs
* Catalog

### Plugins

All I/O implementations are as plugins to the
system, there is no special "included" I/O
that is implemented in a different fashion.

* CSV, JSON, HDF5, other file formats
* Third parties can plug into the system
  following the patterns of the included formats.

