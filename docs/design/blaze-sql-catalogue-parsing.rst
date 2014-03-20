==============================
SQL/YAML Parsing with Blaze
==============================

Goal
====

With a common set (catalogue) of SQL queries the barrier for analysis can be greatly reduced -- the barrier
beginning the database which a user is fetching data from.  Using a catalogue of SQL queries will mean
separating the load between teams of people with SQL experience and those without.  Additionally, developing a
catalogue can mean that groups with limited SQL experience can leverage the knowledge of single person without
much effort.  Blaze should have a standard format for developing such a catalogue.

Background
==========

Blaze already has support for the concept of a **catalogue** and currently loads config options from a yaml file.
We therefore propose a YAML format for defining SQL queries, possible conditionals for those queries, and a possible
typing system for valid parameteres

Requirements
============

Blaze should should parse the YAML and expose the catalogue through a URL.  Arguments listed as **conditionals**
should be available for use by the user in **where** clause of the loaded blaze array

Example Use
===========

A simple use case where for defining and loading a YAML file containing a SQL query and conditional arguments
defined::


    eod_query = """\
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
    """

    bsql_cat = blaze.sql_catalogue_load(path_to_catalogue,db_conn)
    barr = bsql_cat['/SQL_CATALOGUE/STOCKS/eod_query.yaml']
    barr.where(and_((barr.ticker.in_(['A',B','C'],
                     barr.date.between_('2001-01-01','2004-01-01')
                   )
              )
