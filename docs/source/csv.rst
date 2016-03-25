===============================
Tips for working with CSV files
===============================

How to
------

Typically one provides a csv filename to the ``data`` constructor like so

.. code-block:: python

   >>> d = data('myfile.csv')  # doctest: +SKIP

GZip extensions or collections of csv files are handled in the same manner.

.. code-block:: python

   >>> d = data('myfile-2014-01-*.csv.gz')  # doctest: +SKIP

In the case of collections of CSV files the files are sorted by filename and
then considered to be concatenated into a single table.


How does it work?
-----------------

Blaze primarily relies on Pandas to parse CSV files into DataFrames.  In the
case of large CSV files it may parse them into several DataFrames and then use
techniques laid out in the :doc:`ooc` section.


What to do when things go wrong
-------------------------------

The same thing that makes CSV files so popular with humans, simple
readability/writability, makes them challenging for computers to reason about
robustly.

Interacting with CSV files often breaks down in one of two ways

1.  We incorrectly guess the dialect of the CSV file (e.g. wrong delimiter, presence or absense of a header, ...)
2.  We incorrectly guess the type of a column with the CSV file (e.g. an integer column turns out to have floats in it)

Because Blaze operates in a lazy way, giving you access to large CSV files
without reading the entire file into memory it is forced to do some guesswork.
By default it guesses the dialect and types on the first few hundred lines of
text.  When this guesswork fails the user must supply additional information.


Correcting CSV Dialects
~~~~~~~~~~~~~~~~~~~~~~~

In the first case of incorrect guessing of CSV dialect (e.g. delimiter) Blaze
respects and passes through all keyword arguments to `pandas.read_csv`_.


.. note::

   In the case of a CSV file with all string data, you must pass the
   ``has_header=True`` argument if the first row is the header row.



Correcting Column Types
~~~~~~~~~~~~~~~~~~~~~~~

In the second case of incorrect guessing of column types Blaze accepts a
:doc:`datashape` as an additional keyword argument.  Common practice is to create a
``data`` object around a csv file, ask for its datashape, tweak that datashape
and then recreate the data object.

.. code-block:: python

   >>> d = data('myfile.csv')  # doctest: +SKIP
   >>> d  # doctest: +SKIP
   Exception: Integer column has NA values

   >>> d.dshape  # Perhaps that integer column should be a float  # doctest: +SKIP
   dshape("var * {name: string, amount: int64}")

   # <Copy-Paste>
   >>> ds = dshape("var * {name: string, amount: float64}")  # change int to float  # doctest: +SKIP

   >>> d = data('myfile.csv', dshape=ds)  # doctest: +SKIP


Migrate to Binary Storage Formats
---------------------------------

If you plan to reuse the same CSV files many times it may make sense to convert
them to an efficient binary store like HDF5 (common) or BColz (less common but
faster).  These storage formats provide better performance on your data and
also avoid the ambiguity that surrounds CSV files.

One can migrate from CSV files to a binary storage format using the ``odo``
function.

.. code-block:: python

   >>> from odo import odo
   >>> odo('myfiles-*.csv', 'myfile.bcolz')  # doctest: +SKIP

   # or

   >>> odo('myfiles-*.csv', 'myfile.hdf5::/mydataset')  # doctest: +SKIP

   # or

   >>> odo('myfiles-*.csv', 'sqlite:///mydb.db::mytable')  # doctest: +SKIP

When migrating from a loosely formatted system like CSV to a more strict system
like HDF5 or BColz there are a few things to keep in mind

1.  Neither supports variable length strings well
2.  But each supports fixed-length strings well and supports compression to
    cover up overly large/wasteful fixed-lengths
3.  HDF5 does not support datetimes well but can easily encode datetimes as
    strings
4.  BColz is a column store, offering much better performance on tables with
    many columns
5.  HDF5 is a standard technology with excellent library support outside of
    the Python ecosystem

To ensure that you encode your dataset appropriately we recommend passing a
datashape explicitly.  As in our previous example this can often be done by
editing automatically generated datashapes

.. code-block:: python

   >>> d = data('myfile.csv')  # doctest: +SKIP
   >>> d.dshape  # doctest: +SKIP
   dshape("var * {name: string, amount: int64}")

   # <Copy-Paste>
   >>> ds = dshape("var * {name: string[20, 'ascii'], amount: float64}")  # doctest: +SKIP

   >>> from odo import odo
   >>> odo('myfiles-*.csv', 'myfile.bcolz', dshape=ds)  # doctest: +SKIP

Providing a datashape removes data type ambiguity from the transfer.

.. _`pandas.read_csv`: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.io.parsers.read_csv.html
