Blaze Catalog
=============

The Blaze catalog is a component within Blaze which
provides a mechanism for finding arrays available
locally, within a Blaze Cluster, or on the internet.
Each array is identified in a local, cluster, or
global namespace by a URI.

Closely connected is the [Blaze Server](blaze-server.md),
which provides access to the catalog and other
Blaze functionality via a web service.

The catalog is where most of the functionality
in the [Array Management
Repo](https://github.com/ContinuumIO/ArrayManagement)
goes.

Use Cases
---------

### Local Ad Hoc Data Analysis

A common pattern in analyzing data sets is to import
data from some files or other inputs, do some cleaning
on the data, then proceed with the analysis using scipy,
pandas, and other tools. The catalog can provide a way to
encapsulate the data import/cleaning.

In this case, extending and accessing the local
catalog without relying on any separate server
processes is important. It should be easy to add an
array to the server, possibly with as little work as
dropping a folder containing the needed data and
metadata in a specific location.

Additionally, it should be possible to access the
data in many common forms, including blaze, dynd,
numpy, pandas, and raw python objects.

Desired Features
----------------

### Data Abstraction	

* Stores arrays with an associated Blaze Datashape.
* Associate arbitrary key/value metadata with each array.
* Represent data directly, or in the form
  of deferred expressions and queries.

### Data Import

* Way to add data from local files (csv, tsv,
  json, netcdf, hdf5, etc) to the catalog.
* Ability to insert data cleaning operations
  that work in a streaming fashion, per-element.
* Ability to insert data cleaning operations
  that work with the whole array at once, basically
  an arbitrary Python function manipulating the data.

### Data Access

* Retrieve the datashape of an array without getting
  its data.
* Retrieve additional metadata ({string : object}
  dictionary, basically) about the array.
* Retrieve data from an array in the catalog using
  slices to restrict which data is read.
* Retrieve data from an array in the catalog filtered
  using a expression.
* Get the array as a blaze array, numpy array,
  pandas array, or dynd array.

### Caching

* Caching of array data on disk in an efficient
  binary form, in cases where the original form
  is not efficient.
* Caching of array data in memory, with user-configurable
  memory limits.

### Implementation

* Implemented as pure Python code, using other Blaze
  components.

Interface
---------

### Importing Arrays Into Blaze

The Blaze Catalog is accessed from the `blaze.catalog`
namespace. Catalog entries live in a directory
structure, and may contain array data directly
or refer to it elsewhere. The configuration is
stored in the directory ~/.blaze/config.yaml,
which might look something like this:

```
catalog:
  local: ~/Arrays
  cluster: D:/BlazeCluster
  cache: D:/ArrayCache
  cachelimit: 20GB
```

To add an array to the catalog, one adds data
files or .yaml files describing how to interpret
the data to the catalog directories, describing the
data and how to import it.

One may, for example, drop a .npy or .csv into the
`~/Arrays` directory, and immediately have it be
picked up by Blaze. If the input file is not in
an efficient format, Blaze will cache it either
by making a copy of the data in an efficient format,
or an creating an index of the original file. 

In the case of a .yaml file, more control is afforded
over how the data gets loaded. For example with
a .csv file input, one can indicate whether there is
a header column, and provide a datashape to indicate
the exact types of the values, instead of relying
on heuristics to determine it.

### Accessing Arrays From Blaze

There are two mechanisms for accessing arrays
from the catalog. One is directly using the URL
of an array, with the get function.

```python
from blaze import catalog as cat

loc = cat.get('local://myarray')
clust = cat.get('cluster://distarray')
glob = cat.get('https://blaze.continuum.io/samples/tycho2')
```

These gives back a catalog entry, which provides the
datashape, as well as methods to retrieve the whole
or part of the array into memory.

The other mechanism is to get a proxy object for
the namespace, which gives access to the catalog
entries as properties.

```python
from blaze import catalog as cat

loc = cat.local.myarray
clust = cat.cluster.distarray
```

These are the same two catalog entries, but accessed
from the two namespace proxy objects.