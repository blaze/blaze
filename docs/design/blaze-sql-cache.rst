==============================
Caching SQL queries with Blaze
==============================

Goal
====

Querying SQL databases can be costly if they receive many requests per
second so they can be be overloaded.  Using a local cache can free the
database from replying to repetitive queries or queries showing some
overlap, allowing a much better response times for queries that have
not been done before.

Background
==========

Blaze already has support for different I/O backends that can
efficiently deal with large datasets, being the most interesting ones
BLZ and HDF5.  BLZ can be interesting because it brings storage in
either a row-wise (barray) or a column-wise (btable) fashion.  An
additional advantage of these is that they support efficient appends
which will be very important for implementing a cache.

Blaze also has provision for a SQL backend that could be used to query
remote databases and use the HDF5 data descriptor to store the data
locally.  So, users could send their queries in pure SQL and the
system would cache them in HDF5 for faster retrieval.  HDF5 also
supports on-the-flight compression/decompression, so it can cache
large amounts of SQL data with relatively low disk consumption (and
requiring much less space than a traditional database for sure).

Requirements
============

One should devise a way to express SQL queries to the relational
database in terms of a customizable form so that the user can ask for
more data but varying a fixed set of parameters (typically the limits
for the query).

Also, the requirement of being able to query the cache from other
languages like R, makes HDF5 the best option for this task.

The cache system will figure out the number of fields and its types so
that it will create an HDF5 table on-disk for caching the results
locally.  The cached data will be indexed using the fields that are
typically used to filter the results.  Blaze should be improved in
order to read only the data that is necessary and indexes will be
instrumental for doing this.

This cache subsystem should be part of Blaze in the same way than the
catalog, and could be used as a general cache system for querying
remote datasets and caching results.  One should devise a way to clean
the cache when no more space is available, but a LRU system should
suffice initially.

Implementation outline
======================

The details of where the cache should live, its size and other
parameters would be implemented via the Blaze catalog::

  cache = blaze.catalog.activate('my_cache')

where 'my_cache.cache' is a YAML file with the different properties
for the cache.  An example of it could be::

   type: cache
   import: {
       datapath: /caches/my_cache.h5
       storage_type: hdf5
   }
   maxsize: 1 GB  # max size for the cache
   evict_policy: LRU  # could be None, for not updating anything 

With that, one can add different queries for getting into the cache.
For example::

  query1 = "select * from table1 where date between %s and %s"
  cache.add('query1', query1, db_connector1)

  query2 = "select * from table2 where temp between %s and %s"
  cache.add('query2', query2, db_connector2)

And caches for query forms can be explictly removed too::

  cache.remove('query1')
  cache.remove('query2')

As well as stopping this specific catalog cache::

  cache.deactivate()

Now, the cache subsystem in Blaze will fetch the metainformation for
the different tables in the queries (table1, table2) by using the
different connectors ('db_connector1', 'db_connector2'), and will
create different data containers inside the cache catalog.

Once the cache containers are created, Blaze will setup a hook on the
appropriate backend (SQL in the examples above) so that, whenever it
does a new query with the same form than one of the cached ones
('query1'. 'query2'), the result will get stored into the corresponding
cache container.

The next time that a query matches some query form, the new range will
be compared against the ranges that are stitting in the corresponding
data containers ('query1', 'query2') in cache.  In case the range
falls into the ranges in cache (either completely or partially, see
"Use cases" sections), the data will be retrieved from cache instead
of issuing the query against the original DB connectors and returned
to the user.

Use cases
=========

A very simple use case is when we do a query based on a date range::

  # Setup the caching on a query form
  cache = blaze.catalog.activate('my_cache')
  query = "select * from mytable where date between %s and %s"
  cache.add('query', query, db_connector1)

  # Do the query and cache the result
  a = blaze.array_from_sql(query, ('2010', '2012'), db_connector)

  # Should hit catalog cache
  b = blaze.array_from_sql(query, ('2010', '2011'), db_connector)

  # We are done with caching with this specific query form
  cache.remove('query')

  # We are done with 'my_cache' completely
  cache.deactivate()

Note how the cache can be activated and deactivated by user request,
both in a query form or a specific cache catalog.  This is important
because sometimes the user won't want to use the caching feature.

A somewhat more complex use case (range overlap)::

  # Do the query and cache the result
  a = blaze.array_from_sql(query, ('Oct-2010', 'May-2011'), db_connector)

  # Do the query and cache the result
  b = blaze.array_from_sql(query, ('Feb-2011', 'Nov-2012'), db_connector)

  # Should hit catalog cache for the whole range
  c = blaze.array_from_sql(query, ('2010', '2012'), db_connector)

In this case, the cache is made of overlapping queries (a and b) that
are stored and then retrieved to form a bigger date range (c).

The 'challenge' in this second case is to recognize overlapping ranges
and not retrieve duplicates during the cached query.  Of course, it
would be even more optimal if duplicates are not stored in the cache
at all.

Another example including 'holes' in ranges::

  # Do the query and cache the result
  a = blaze.array_from_sql(query % ('Oct-2010', 'Feb-2011'), db_connector)

  # Do the query and cache the result
  b = blaze.array_from_sql(query % ('May-2011', 'Nov-2012'), db_connector)

  # Should hit catalog cache
  c = blaze.array_from_sql(query % ('2010', '2012'), db_connector)

In this case, one could take a couple of approaches:

1) Use the cache and fill the holes with actual queries
2) Do not use the cache at all

It seems like 1) should be more efficient, but sometimes not using the
query and asking for the complete range to the database would be more
efficient.  Maybe some heuristics would be nice for implementing 1).
