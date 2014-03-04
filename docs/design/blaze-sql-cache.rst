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

Design
======

Blaze already has support for different I/O backends that can
efficiently deal with large datasets, being the most interesting ones
BLZ and HDF5.  BLZ can be interesting because it brings storage in
either a row-wise (barray) or a column-wise (btable) fashion.
However, the requirement of being able to query the cache from other
languages like R, makes HDF5 the best option for this task.

Blaze also has provision for a SQL backend that could be used to query
remote databases and use the HDF5 data descriptor to store the data
locally.  So, users could send their queries in pure SQL and the
system would cache them in HDF5 for faster retrieval.  HDF5 also
supports on-the-flight compression/decompression, so it can cache
large amounts of SQL data with relatively low disk consumption (and
requiring much less space than a traditional database for sure).

Implementation
==============

One should devise a way to express SQL queries to the relational
database in terms of a customizable form so that the user can ask for
more data but varying a fixed set of parameters (typically the limits
for the query).

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

Use cases
=========

A very simple use case is when we do a query based on a date range::

  # Setup the caching on a query form
  query = "select * from mytable where date between %s and %s"
  blaze.cache.activate(query, db_connector)

  # Do the query and cache the result
  a = blaze.array_from_sql(query % ('2010', '2012'), db_connector)

  # Should hit catalog cache
  b = blaze.array_from_sql(query % ('2010', '2011'), db_connector)

  # We are done with caching
  blaze.cache.deactivate()

Please note how the cache can be activated and deactivated by user
request.  This is important because sometimes the user won't like to
use the caching feature.

A somewhat more complex use case (range overlap)::

  # Do the query and cache the result
  a = blaze.array_from_sql(query % ('Oct-2010', 'May-2011'), db_connector)

  # Do the query and cache the result
  b = blaze.array_from_sql(query % ('Feb-2011', 'Nov-2012'), db_connector)

  # Should hit catalog cache for the whole range
  c = blaze.array_from_sql(query % ('2010', '2012'), db_connector)

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
