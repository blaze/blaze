Blaze Catalog
=============

The Blaze catalog is a component within Blaze which
provides a mechanism for finding arrays available
within the Blaze Cluster the machine is a part of.
This means locally on the current machine, if no
cluster has been explicitly configured, or within
the directory structure shared across the cluster
if one has.

Closely connected is the [Blaze Server](blaze-server.md),
which provides access to the catalog and other
Blaze functionality via a web service, and is how
members of a cluster communicate with each other.

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
processes is important. It should be straightforward
to add an array to the server, as easy as loading
data using pandas, for example.

Additionally, it should be possible to access the
data in many common forms, including blaze, dynd,
numpy, pandas, and raw python objects (e.g. lists
of dictionaries or tuples). This functionality is
not specific to the catalog, rather a feature to
be supported by any Blaze array.

Desired Features
----------------

### Data Abstraction	

* Within a configured cluster, a single rooted
  namespace containing all persisted Blaze Arrays
  within the cluster.
  * Still want to be able to use URLs from other
    Blaze servers/catalogs directly without an import
    step.
* A catalog entry includes the following associated data.
  * Whether it produces a blaze.Array or a blaze.Table.
  * DataShape of the Array. Note that this scales from
    single values like a boolean or string, up through
    tables and multi-dimensional arrays.
  * The information needed to retrieve the Array's
    data. If a remote data source or file location
    is known to be fixed, this can be lazily loaded.
    Alternatively, an import step could snapshot the
    data when the it may change or disappear.
  * In the case of a deferred expression, the input
    arrays need to be tracked as a dependency graph,
    e.g. affecting how user management tools of the
    catalog warn about deleting arrays which others
    depend upon.
  * User-specified metadata about the array.
    Likely JSON or JSON-like key/value pairs of data.
* Local per-machine temporary namespace, `'/tmp'`.
* Per-user namespace, `'/home/<username>'`.

### Data Import

* Way to add data from local files (csv, tsv,
  other delimited text files, npy, json, netcdf, hdf5,
  etc) to the catalog.
* Data which is served publically via
  a [Blaze Server](blaze-server.md) should be
  importable into the cluster namespace with only
  the source array URL and the destination catalog
  path.
* Ability to insert data cleaning operations
  that work in a streaming fashion, per-element.
* Ability to insert data cleaning operations
  that work with the whole array at once, basically
  an arbitrary Python function manipulating the data.
* TODO: Add error handling info (mostly higher level).
* TODO: Cache invalidation when data changes via
        push, reevaluation in a lazy way via pull.

Peter - people like in ETL tools, rich error handling.

### Data Access

* Retrieve an Array object for any catalog entry
  with a convenient syntax.
    * `blaze.get("/path/to/array")`,
      `blaze.get("~/userarray")`
* Access permissions is not part of the catalog,
  just using the permissions of underlying FS for now.
* Create blaze command line/a series of ipython
  magics for exploring the catalog structure.

### Caching

* Caching of array data on disk in an efficient
  binary form, in cases where the original form
  is not efficient.
* Caching of array data in memory, with user-configurable
  memory limits.
* The cache needs tooling, users should be able to
  control how much space is in the cache, see the
  usage, etc.
* Caching is implicit, but with an explicit way
  to skip the cache if you'll just use data once.

### Implementation

* Implemented as pure Python code, using other Blaze
  components.
* Initial implementation is close to the style of
  Hugo's ArrayManagement repo, not using a sqlite
  database for now.

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
from the catalog. One is directly using the path
of an array, with the get function.

```python
from blaze import catalog as cat

userarray = cat.get('~/myarray')
userarray2 = cat.get('/home/<username>/myarray2')
sharedarray = cat.get('/tycho2')
```

These gives back Blaze Array objects.
