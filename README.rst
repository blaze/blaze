.. image:: https://raw.github.com/blaze/blaze/master/docs/source/svg/blaze_med.png
   :align: center

|Build Status| |Coverage Status| |Join the chat at
https://gitter.im/blaze/blaze|

**Blaze** translates a subset of modified NumPy and Pandas-like syntax
to databases and other computing systems. Blaze allows Python users a
familiar interface to query data living in other data storage systems.

Example
=======

We point blaze to a simple dataset in a foreign database (PostgreSQL).
Instantly we see results as we would see them in a Pandas DataFrame.

.. code:: python

    >>> import blaze as bz
    >>> iris = bz.Data('postgresql://localhost::iris')
    >>> iris
        sepal_length  sepal_width  petal_length  petal_width      species
    0            5.1          3.5           1.4          0.2  Iris-setosa
    1            4.9          3.0           1.4          0.2  Iris-setosa
    2            4.7          3.2           1.3          0.2  Iris-setosa
    3            4.6          3.1           1.5          0.2  Iris-setosa

These results occur immediately. Blaze does not pull data out of
Postgres, instead it translates your Python commands into SQL (or
others.)

.. code:: python

    >>> iris.species.distinct()
               species
    0      Iris-setosa
    1  Iris-versicolor
    2   Iris-virginica

    >>> bz.by(iris.species, smallest=iris.petal_length.min(),
    ...                      largest=iris.petal_length.max())
               species  largest  smallest
    0      Iris-setosa      1.9       1.0
    1  Iris-versicolor      5.1       3.0
    2   Iris-virginica      6.9       4.5

This same example would have worked with a wide range of databases,
on-disk text or binary files, or remote data.

What Blaze is not
=================

Blaze does not perform computation. It relies on other systems like SQL,
Spark, or Pandas to do the actual number crunching. It is not a
replacement for any of these systems.

Blaze does not implement the entire NumPy/Pandas API, nor does it
interact with libraries intended to work with NumPy/Pandas. This is the
cost of using more and larger data systems.

Blaze is a good way to inspect data living in a large database, perform
a small but powerful set of operations to query that data, and then
transform your results into a format suitable for your favorite Python
tools.

In the Abstract
===============

Blaze separates the computations that we want to perform:

.. code:: python

    >>> accounts = Symbol('accounts', 'var * {id: int, name: string, amount: int}')

    >>> deadbeats = accounts[accounts.amount < 0].name

From the representation of data

.. code:: python

    >>> L = [[1, 'Alice',   100],
    ...      [2, 'Bob',    -200],
    ...      [3, 'Charlie', 300],
    ...      [4, 'Denis',   400],
    ...      [5, 'Edith',  -500]]

Blaze enables users to solve data-oriented problems

.. code:: python

    >>> list(compute(deadbeats, L))
    ['Bob', 'Edith']

But the separation of expression from data allows us to switch between
different backends.

Here we solve the same problem using Pandas instead of Pure Python.

.. code:: python

    >>> df = DataFrame(L, columns=['id', 'name', 'amount'])

    >>> compute(deadbeats, df)
    1      Bob
    4    Edith
    Name: name, dtype: object

Blaze doesn't compute these results, Blaze intelligently drives other
projects to compute them instead. These projects range from simple Pure
Python iterators to powerful distributed Spark clusters. Blaze is built
to be extended to new systems as they evolve.

Getting Started
===============

Blaze is available on conda or on PyPI

::

    conda install blaze
    pip install blaze

Development builds are accessible

::

    conda install blaze -c blaze
    pip install http://github.com/blaze/blaze --upgrade

You may want to view `the docs <http://blaze.pydata.org>`__, `the
tutorial <http://github.com/blaze/blaze-tutorial>`__, `some
blogposts <http://continuum.io/blog/tags/blaze>`__, or the `mailing list
archives <https://groups.google.com/a/continuum.io/forum/#!forum/blaze-dev>`__.


Development setup
=================

The quickest way to install all Blaze dependencies with ``conda`` is as
follows

::

    conda install blaze spark -c blaze -c anaconda-cluster -y
    conda remove odo blaze blaze-core datashape -y

After running these commands, clone ``odo``, ``blaze``, and ``datashape`` from
GitHub directly.  These three projects release together.  Run ``python setup.py
develop`` to make development installations of each.


License
=======

Released under BSD license. See `LICENSE.txt <LICENSE.txt>`__ for
details.

Blaze development is sponsored by Continuum Analytics.

.. |Build Status| image:: https://travis-ci.org/blaze/blaze.png
   :target: https://travis-ci.org/blaze/blaze
.. |Coverage Status| image:: https://coveralls.io/repos/blaze/blaze/badge.png
   :target: https://coveralls.io/r/blaze/blaze
.. |Join the chat at https://gitter.im/blaze/blaze| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/blaze/blaze?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
