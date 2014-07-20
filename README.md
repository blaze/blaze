<p align="center" style="padding: 20px">
<img src="https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/blaze_med.png">
</p>

**Blaze** extends the usability of NumPy and Pandas to distributed and
out-of-core computing.  Blaze provides an interface similar to that of the
NumPy ND-Array or Pandas DataFrame but maps these familiar interfacess onto a
variety of other computational engines like Postgres or Spark.

### Example

Blaze separates the computations that we want to perform:

```Python
>>> accounts = TableSymbol('accounts',
...                        schema='{id: int, name: string, amount: int}')

>>> deadbeats = accounts[accounts['amount'] < 0]['name']
```

From the representation of that data

```Python
>>> L = [[1, 'Alice',   100],
...      [2, 'Bob',    -200],
...      [3, 'Charlie', 300],
...      [4, 'Denis',   400],
...      [5, 'Edith',  -500]]
```

Blaze enables users to solve data-oriented problems

```Python
>>> list(compute(deadbeats, L))
['Bob', 'Edith']
```

The separation of expression from data allows users to switch seemlessly
between different backends.  Here we solve the same problem using Pandas
instead of Pure Python.

```Python
>>> df = DataFrame(L, columns=['id', 'name', 'amount'])

>>> compute(deadbeats, df)

```

Blaze allows us to write down what we want to compute abstractly.  Blaze also
knows how to drive other systems to compute what was asked.  These systems can
range from simple Pure Python iterators to powerful distributed Spark clusters.
Blaze doesn't compute your answer, Blaze intelligently drives other projects to
do the computation.


### Useful Abstractions
<p align="center" style="padding: 20px">
<img src="https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/numpy_plus.png">
</p>

Blaze includes a rich set of computational and data primitives useful in
building and communicating between computational systems.  Blaze primitives can
help with consistent and robust data migration, as well as remote execution.

<p align="center" style="padding: 20px">
<img src="https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/codepush.png">
</p>

Blaze aims to be a foundational project allowing many different users of
other PyData projects (Pandas, Theano, Numba, SciPy, Scikit-Learn)
to interoperate at the application level and at the library level with
the goal of being able to to lift their existing functionality into a
distributed context.

<p align="center" style="padding: 20px">
<img src="https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/sources.png">
</p>


Getting Started
---------------

Again, Blaze is in development.  We reserve the right to break the API.

We are currently looking for patient users and creative developers.  If you
have a problem that can use Blaze or if you have a computation backend or data
file format that would like integrate into Blaze then please contact the
[Mailing list]() or [Install the development version of Blaze]() and let us
know about your experience.

Source code for the latest development version of blaze can
be obtained [from Github](https://github.com/ContinuumIO/blaze).

Development installation instructions available [here]()


Installing from Source
----------------------

Install all the pre-requisites using conda or another mechanism,
then run:

```bash
$ python setup.py install
```

Contributing
------------

Anyone wishing to discuss on Blaze should join the
[blaze-dev](https://groups.google.com/a/continuum.io/forum/#!forum/blaze-dev)
mailing list. To get started contributing, read through the
[Developer Workflow](docs/source/dev_workflow.rst) documentation.


Documentation
-------------

Documentation is available at
[blaze.pydata.org/docs/dev/](http://blaze.pydata.org/docs/dev/)


License
-------

Blaze development is sponsored by Continuum Analytics.

Released under BSD license. See [LICENSE.txt](LICENSE.txt) for details.
