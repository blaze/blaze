==========================
SQL To Blaze YAML Handling
==========================

Goal
====

With a common set (catalog) of SQL queries the barrier for analysis can be greatly reduced -- the barrier
being the database which a user is fetching data from.  Using a catalog of SQL queries will mean
separating the load between teams of people with SQL experience and those without.  Additionally, developing a
catalog can mean that groups with limited SQL experience can leverage the knowledge of single person without
much effort.  Blaze should have a standard format for developing such a catalog.

Background
==========

Blaze already has support for the concept of a **catalog** and currently loads arrays from a yaml file with suffix:
**.array**.  We therefore propose a YAML format for defining SQL queries, conditionals for those queries, and
a possible typing system for parameter validation.

Requirements
============

Blaze should should parse the YAML file and expose the catalog through a URL.  Arguments listed as **conditionals**
should be available for use by the user in **where** clauses of the loaded blaze array.  Blaze should be understand
and manage a hierarchical folder structure of ~10K+ **.array** SQL files -- where each file is can either be an
individual SQL call or a collection of SQL call

In ArrayManagement, we can using the following structure and refer to queries/arrays as URLs relative to a common
root::

    SQL_catalog
    |-- Fundamentals
    |    |-- DB1
    |    |    |-- file1_db1.array
    |    |    |-- file2_db1.array
    |    +-- DB2
    |         |-- file.array
    +-- End_of_Day
        |-- ohlc.array
        |-- marketcap.array

    client['/SQL_catalog/End_of_Day/ohlc.array']
    #or
    client['/SQL_catalog/Fundamentals/file1_db1.array']

We additionally require blaze to parse YAML of the following form::


    #eod_query.yaml

    stock_query:
      type: 'conditional'
      conditionals:
          - date
          - ticker

      query: >
          SELECT stocks.ticker, s
          tock_hist.c,
          stock_hist.o,
          stock_hist.date,
          FROM     stocks
          JOIN     stock_hist
              ON  stocks.sec_id = stock_hist.sec_id



The above is a valid file in master of ArrayManagement.  Because queries and conditional parameters are separated, and
because ArrayManagement makes use of SQLAlchemy, we can now attach arbitrary sub-selections based on the
conditionals provided in the YAML file::

    arr = client['/SQL_catalog/STOCKS/eod_query.yaml']
    arr.select(and_((arr.ticker.in_(['A',AA','AAPL'],
                     arr.date.between_('2001-01-01','2004-01-01')
                   )
              )

When an ArrayManagement client is handed a valid URL, the file is parsed, the conditionals are loaded as sqlalchemy
column attributes, and these attributes can now be used with conditional logic **and_**, **or_**, **in_**,
**equality (==)**, etc.

We require Blaze to offer comparable functionality:

- Load SQL data from a standardized YAML file:
- Expose query as a blaze array for subselection
- subselection methods should leverage SQLAlchemy

