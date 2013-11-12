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
  json, netcdf, hdf4, etc) to the catalog.
* Ability to insert data cleaning operations
  that work in a streaming fashion, per-element.
* Ability to insert data cleaning operations
  that work with the whole array at once, basically
  an arbitrary Python function manipulating the data.

### Data Access

* Retrieve data from an array in the catalog using
  slices to restrict which data is read.
* Retrieve data from an array in the catalog filtered
  using a expression.

### Caching

* Caching of array data on disk in an efficient
  binary form, in cases where the original form
  is not efficient.
* Caching of array data in memory, with user-configurable
  memory limits.

### Implementation

* Implemented as pure Python code, using other Blaze
  components.
