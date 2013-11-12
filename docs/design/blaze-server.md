Blaze Server
============

The Blaze server is a component of Blaze which
exposes array data from a local
[Blaze Catalog](blaze-catalog.md) to the network as
a web service. Its main goals are to provide a way
to serve array and table data in a rich way, and to
expose a blaze node's computation and storage to
a cluster.

Requirements
------------

### Serving Rich Data

* Expose the Blaze Catalog defined on the local server.
* Serve data in JSON, for clients that do not
  understand the binary form.
* Serve data in binary form. This requires definition
  of a serialization format for datashaped data.
* Server-side slicing of datasets.
* Server-side filtering of datasets.
* Server-side computed columns.
* Should support using POST for specifying all
  the data, so that queries can be done without
  leaving behind traces in web caching servers, etc.

### Supporting Blaze Cluster

* Support embarrassingly parallel computation
  via numpy-style broadcasting/loop fusion of deferred
  computations.
* Head node and compute node. Design needed here,
  like choosing a user-configured head node (i.e. configured
  to be optimal for the scheduling tasks)
  or arbitrarily selected, to avoid the single
  failure point.
* Ability to define a cluster, with a shared namespace
  for catalog access.
* Support a "compute context" which can receive
  code to execute on the local data.

Development Roadmap
-------------------

The first priority is building the capacity to serve
rich data using Blaze. The target client for this is
[Bokeh](https://github.com/ContinuumIO/Bokeh), which
is defining a Bokeh Server. The Bokeh Server would include
the Blaze Server as a subcomponent, providing direct access
to read the underlying data driving Bokeh plots.

Developing the cluster functionality is scheduled after
the catalog is functioning well for a single server
of rich data.

Inspiration from OPeNDAP/DAP
----------------------------

One of the inspirations for the Blaze Server is the
[OPeNDAP project](http://opendap.org/), which describes
the following reasons for its use:

* Allows you to access data over the internet.
* [DAP 2.0](https://earthdata.nasa.gov/our-community/esdswg/standards-process-spg/rfc/esds-rfc-004-dap-20) is a NASA community standard.
* Data is stored and transmitted in binary form.
* OPeNDAP provides sophisticated sub-sampling capabilities.

In DAP, the [DDS Format](http://docs.opendap.org/index.php/UserGuideDataModel)
is analogous to Blaze Datashape. The format for
storage and transmission of data is defined in
terms of [XDR](https://tools.ietf.org/html/rfc4506).
There is a Python implementation called
[PyDAP](http://www.pydap.org/).

Some limitations of DAP:

* No boolean, 8-bit integers, datetimes, enumerations,
  categorical types.
* The DAP v4 draft on the OPeNDAP site is from 2004,
  cannot tell if progress is being made.

