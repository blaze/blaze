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

  cache_desc = blaze.catalog.activate('my_cache')

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

  sql_arr1 = blaze.array("select * from table1", db_conn1)
  sql_arr1.activate_cache(cache_desc)

  sql_arr2 = blaze.array("select * from table2", db_conn2)
  sql_arr2.activate_cache(cache_desc)

Now, the cache subsystem in Blaze will fetch the metainformation for
the different tables in the queries (table1, table2) by using the
different connectors (db_conn1, db_conn2), and will store the info
about the data to be retrieved (the column names and types basically)
inside the cache catalog.

After this, Blaze will setup a hook in the appropriate backend (SQL in
the examples above) so that, whenever it does a new query using the
deferred arrays above, the result will get stored into the
corresponding cache container.  For example::

  a = sql_arr1.where("2010 < date < 2011")

will use the initial query for `sql_arr1` and will build the next
one::

  select * from table1 where date between 2010 and 2011

and the result will be stored in the underlying cache, as well as
returned to the user.

The next time that the deferred array is queried, the new range will
be compared against the ranges that are stitting in the corresponding
data containers in cache.  In case the range falls into the ranges in
cache (either completely or partially, see "Use cases" sections), the
data will be retrieved from cache instead of issuing the query against
the original database connectors and returned to the user.
 
Caches for deferred arrays can be explictly removed too::

  sql_arr1.deactivate_cache()
  sql_arr2.deactivate_cache()

Stopping caching all the queries in a specific catalog cache can be
done with::

  cache_desc.deactivate()


Use cases
=========

A very simple use case is when we do a query based on a date range::

  # Setup the caching on a deferred array
  cache_desc = blaze.catalog.activate('my_cache')
  sql_arr = blaze.array("select * from table1", db_conn)
  sql_arr.activate_cache(cache_desc)

  # Do the query and cache the result
  a = sql_arr.where("2010 < date < 2012")

  # This new query should hit catalog cache
  b = sql_arr.where("2010 < date < 2011")

  # We are done with caching with this specific deferred array
  sql_arr.deactivate_cache()

  # We are done with 'my_cache' completely
  cache_desc.deactivate()

Note how the cache can be activated and deactivated by user request,
both in a deferred array or on a specific cache catalog.  This is
important because sometimes the user won't want to use the caching
feature (there can be fundamental reasons for that).

A somewhat more complex use case (range overlap)::

  # Do the query and cache the result
  a = sql_arr.where("Oct-2010 < date < May-2011")

  # Do the query and cache the result
  b = sql_arr.where("Feb-2011 < date < Nov-2012")

  # Should hit catalog cache for the whole range
  c = sql_arr.where("Nov-2010 < date < Sep-2012")

In this case, the cache is made of overlapping queries (a and b) that
are stored and then retrieved to form a bigger date range (c).

The 'challenge' in this second case is to recognize overlapping ranges
and not retrieve duplicates during the cached query.  Of course, it
would be even more optimal if duplicates are not stored in the cache
at all.

Another example including 'holes' in ranges::

  # Do the query and cache the result
  a = sql_arr.where("Oct-2010 < date < Feb-2011")

  # Do the query and cache the result
  b = sql_arr.where("May-2011 < date < Nov-2012")

  # Should hit catalog cache in some date ranges
  c = sql_arr.where("Oct-2010 < date < Nov-2012")

In this case, one could take a couple of approaches:

1) Use the cache and fill the holes with actual queries
2) Do not use the cache at all

It seems like case 1 should be more efficient, but sometimes not using
the query and asking for the complete range to the database would be
faster.  Maybe some heuristics would be nice for implementing case 1.

Complete example
================

Here it is a complete example on how the cache should work::

  import blaze
  import pyodbc as db

  # The data for the SQL table
  data = [
      (2010-10-10, "hello", 2.1),
      (2011-11-11, "world", 4.2),
      (2012-12-12, "!",     8.4),
  ]

  # Use ODBC to create a SQLite database in-memory
  conn = db.connect("Driver=SQLite ODBC Driver;")
  c = conn.cursor()
  c.execute("create table my_table (tdate DATE, msg TEXT, price REAL)")
  c.executemany("insert into testtable values (?, ?, ?)""", data)
  conn.commit()
  c.close()

  # Setup the caching on a deferred array
  cache_desc = blaze.catalog.activate('my_cache')
  sql_arr = blaze.array("select * from my_table", conn)
  sql_arr.activate_cache(cache_desc)

  # Do the query and cache the result
  a = sql_arr.where("2010-12-31 < tdate < 2012-01-01")
  # The line below should print: 'array([2011-11-11, "world", 4.2]))'
  print(a)

  # This new query should hit catalog cache
  b = sql_arr.where("2010-12-31 < tdate < 2012-01-01")
  # The line below should print: 'array([2011-11-11, "world", 4.2]))'
  print(b)

  # We are done with 'my_cache' completely
  cache_desc.deactivate()

