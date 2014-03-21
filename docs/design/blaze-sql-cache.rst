==============================
Caching SQL queries with Blaze
==============================

Goal
====

Querying SQL databases can be costly if they receive many requests per
second so they can be be overloaded.  Using a local cache can free the
database from replying to repetitive queries or queries showing some
overlap, allowing a much better response time for queries that have
been performed before.

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

Additionally, caching on continuous parameters such as datetime should
be handled properly.  That is, if one query is between 2001 and 2008
and a second query is between 2005 and 2012, blaze should only pull
the missing data.

This cache subsystem should be part of Blaze in the same way than the
catalog, and could be used as a general cache system for querying
remote datasets and caching results.  One should devise a way to clean
the cache when no more space is available, but a LRU schema for
evicting old entries should suffice initially.

Implementation outline
======================

The details of where the cache should live, its size and other
parameters would be implemented via the Blaze catalog.  The cache
properties can be expressed in a YAML file like::

  type: cache
  import: {
      # Absolute path
      cachedir: /caches/my_cache.h5
      # Relative path
      #cachedir: my_cache.h5
      storage_type: hdf5
  }
  maxsize: 1 GB  # max size for the cache
  evict_policy: LRU  # could be None, for not updating anything 

Note: If the `cachedir` entry is a relative path, it will be prepended
with the following paths::

  ~/.cache/ContinuumIO/Blaze           # Linux
  ~/Library/Caches/ContinuumIO/Blaze   # Mac OSX
  %APPDATA%\ContinuumIO\Blaze          # Win

With cache configuration in place, one can add different queries for
getting into the cache.  For example::

  sql_arr1 = blaze.array("select * from table1", db_conn1)
  sql_arr2 = blaze.array("select * from table2", db_conn2)

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

will use the initial query for `sql_arr1` and will build and execute
the next one::

  select * from table1 where date between 2010 and 2011

Then, the result will be stored in the underlying cache, as well as
returned to the user.

The next time that the deferred array is queried, the new range will
be compared against the ranges that are sitting in the corresponding
data containers in cache.  In case this range matches into data
already stored in cache (either completely or partially, see "Use
cases" sections), this data will be retrieved from cache instead of
issuing the query against the original database connectors and
returned to the user.
 
Caches for deferred arrays can be explictly disabled::

  sql_arr1.deactivate_cache()
  sql_arr2.deactivate_cache()

and re-enabled::

  sql_arr1.activate_cache()
  sql_arr2.activate_cache()

Also, cache contents can be removed too::

  sql_arr1.clear_cache()
  sql_arr2.clear_cache()

Use cases
=========

A very simple use case is when we do a query based on a date range::

  # Setup the caching on a deferred array
  sql_arr = blaze.array("select * from table1", db_conn)

  # Do the query and cache the result
  a = sql_arr.where("2010 < date < 2012")

  # This new query should hit catalog cache
  b = sql_arr.where("2010 < date < 2011")

  # We are done with caching with this specific deferred array
  sql_arr.deactivate_cache()

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
  sql_arr = blaze.array("select * from my_table", conn)

  # Do the query and cache the result
  a = sql_arr.where("2010-12-31 < tdate < 2012-01-01")
  # The line below should print: 'array([2011-11-11, "world", 4.2]))'
  print(a)

  # This new query should hit catalog cache
  b = sql_arr.where("2010-12-31 < tdate < 2012-01-01")
  # The line below should print: 'array([2011-11-11, "world", 4.2]))'
  print(b)



Advanced Example
================

The following is an example of caching for databases that contain a collection of tables (RDBMSs)::

Starting with two tables:

**STOCKS.TBL**

===================  ===================  ======  ======
max_date             min_date             sec_id  ticker
-------------------  -------------------  ------  ------
2013-08-09 00:00:00  1999-11-19 00:00:00  0       A
2013-08-09 00:00:00  1998-01-05 00:00:00  1       AA
2013-08-09 00:00:00  1998-01-05 00:00:00  2       AAPL
...                  ...                  .       .
===================  ===================  ======  ======

**STOCKS_HIST.TBL**

=================== ======= ======= ======= ======= ============ =======
date                o       h       l       c       v            sec_id
------------------- ------- ------- ------- ------- ------------ -------
1999-11-19 00:00:00 39.8329 39.8885 36.9293 37.6251 11390201.186 0
1999-11-22 00:00:00 38.3208 40.0091 37.1613 39.9442 4654716.475  0
1999-11-23 00:00:00 39.4247 40.4729 37.3375 37.5138 4268902.729  0
...                 ...     ...     ...     ...     ...          ...
=================== ======= ======= ======= ======= ============ =======


Blaze caching should store the expression graph of the query and the data::

  sql = '''select stocks.ticker, stock_hist.c, stock_hist.o, stock_hist.date
  from stocks inner join stock_hist on
  stocks.sec_id = stock_hist.sec_id
  '''

  sql_arr = blaze.array(sql, db_conn)
  sql_arr.where(and_(stocks.ticker.in_(['A',B','C']),
                 stock_hist.date.between_('2001-01-01','2004-01-01')
                 )
             )
  print(sql_sub_arr)

The caching/fetching mechanism should be smart enough to fetch only the diff on the following query::

  sql_arr.where(and_(stocks.ticker.in_(['A',B','E','F]),
                 stock_hist.date.between_('2002-01-01','2005-01-01')
                 )
             )

Notice the **where** clause now contains entities: A,B,E,F and the date range has changed to extend beyond
dates which are in the current cache.  Blaze should should fetch data between 2002-01-01 and 2005-01-01 for
entities E and F, and for entities A and B, fetch data between 2004-01-01 and 2005-01-01.
