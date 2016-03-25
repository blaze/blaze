======================
Out of Core Processing
======================

Blaze includes nascent support for out-of-core processing with Pandas
DataFrames and NumPy NDArrays.  It combines a computationally-rich in-memory
solution (like pandas/numpy) with a computationally-poor out-of-core solution.

How do I use this?
------------------

Naive use of Blaze triggers out-of-core systems automatically when called on
large files.

.. code-block:: python

   >>> d = data('my-small-file.csv')  # doctest: +SKIP
   >>> d.my_column.count()  # Uses Pandas  # doctest: +SKIP

   >>> d = data('my-large-file.csv')  # doctest: +SKIP
   >>> d.my_column.count()  # Uses Chunked Pandas  # doctest: +SKIP

How does it work?
-----------------

Blaze breaks up the data resource into a sequence of chunks.  It pulls one
chunk into memory, operates on it, pulls in the next, etc..  After all chunks
are processed it often has to finalize the computation with another operation
on the intermediate results.

In the example above one might accomplish the computation above, counting the
number of non-null elements, with pure Pandas as follows:

.. code-block:: python

   # Operate on each chunk
   intermediate = []
   for chunk in pd.read_csv('my-large-file.csv', chunksize=1000000):
       intermediate.append(chunk.my_column.count())

   # Finish computation by operating on the intermediate result
   result = sum(intermediate)

This example accomplishes a single computation on the entire dataset, ``d.my_column.count()``, by separating it into two stages

1.  compute ``chunk.my_column.count()`` on each in-memory chunk
2.  compute ``intermediate.sum()`` on the aggregated intermediate results

Blaze figures out this process for you.  The code above only serves as an
example of the kind of thing that Blaze does automatically.  Blaze knows how to
separate a broad range of computations.  Notable exceptions include joins and
sorts.  Blaze does not currently support out-of-core computation on joins and
sorts.

Complex Example
---------------

To investigate further try out the ``split`` function in ``blaze.expr.split``.
It will tell you exactly how Blaze intends to break up your computation.  Here
is a more complex example doing an out-of-core split-apply-combine operation:

.. code-block:: python

   >>> from blaze import *
   >>> from blaze.expr.split import split

   >>> bank = symbol('bank', 'var * {name: string, balance: int}')

   >>> expr = by(bank.name, avg=bank.balance.mean())

   >>> split(bank, expr)  # doctest: +SKIP
   ((chunk,
     by(chunk.name, avg_count=count(chunk.balance),
                    avg_total=sum(chunk.balance))),
   (aggregate,
     by(aggregate.name, avg=(sum(aggregate.avg_total)) /
                             sum(aggregate.avg_count))))

As in the first example this chunked split-apply-combine operation translates
the intended results into two different computations, one to perform on each
in-memory chunk of the data and one to perform on the aggregated results.

Note that you do not need to use ``split`` yourself.  Blaze does this for you
automatically.


Parallel Processing
-------------------

If a data source is easily separable into chunks in a parallel manner then
computation may be accelerated by a parallel map function provided by
the ``multiprocessing`` module (or any similar module).

For example a dataset comprised of many CSV files may be easily split up (one
csv file = one chunk.)  To supply a parallel map function one currently must
use the explicit ``compute`` function.

.. code-block:: python

   >>> d = data('my-many-csv-files-*.csv')  # doctest: +SKIP
   >>> d.my_column.count()  # Single core by default  # doctest: +SKIP
   ...

   >>> import multiprocessing  # doctest: +SKIP
   >>> pool = multiprocessing.Pool(4)  # Four processes  # doctest: +SKIP

   >>> compute(d.my_column.count(), map=pool.map)  # Parallel over four cores  # doctest: +SKIP
   ...

Note that one can only parallelize over datasets that can be easily split in a
non-serial fashion.  In particular one can not parallelize computation over
a single CSV file.  Collections of CSV files and binary storage systems like
HDF5 and BColz all support multiprocessing.


Beyond CSVs
-----------

While pervasive, CSV files may not be the best choice for speedy processing.
Binary storage formats like HDF5 and BColz provide more opportunities for
parallelism and are generally much faster for large datasets.
