<p align="center" style="padding: 20px">
<img src="https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/blaze_med.png">
</p>

**Blaze** extends the usability of NumPy and Pandas to distributed and
out-of-core computing.  Blaze provides an interface similar to that of the
NumPy ND-Array or Pandas DataFrame but maps these familiar interfaces onto a
variety of other computational engines like Postgres or Spark.

Example
-------

Blaze separates the computations that we want to perform:

```Python
>>> accounts = TableSymbol('accounts',
...                        schema='{id: int, name: string, amount: int}')

>>> deadbeats = accounts[accounts['amount'] < 0]['name']
```

From the representation of data

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

But the separation of expression from data allows us to switch between
different backends.

Here we solve the same problem using Pandas instead of Pure Python.

```Python
>>> df = DataFrame(L, columns=['id', 'name', 'amount'])

>>> compute(deadbeats, df)
1      Bob
4    Edith
Name: name, dtype: object
```

Blaze doesn't compute these results, Blaze intelligently drives other projects
to compute them instead.  These projects range from simple Pure Python
iterators to powerful distributed Spark clusters.  Blaze is built to be
extended to new systems as they evolve.


Usable Abstractions
-------------------

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

Development installation instructions available [here](http://blaze.pydata.org/docs/dev/dev_workflow.html#installing-development-blaze).  Quick usage available [here](http://blaze.pydata.org/docs/dev/quickstart.html).

Blaze is in development.  We reserve the right to break the API.

Blaze needs your help.  Blaze needs users with interesting problems.  Blaze
needs developers with expertise in new data formats and computational backends.
Blaze needs core developers to tie everything together.  Please e-mail the
[Mailing list](mailto:blaze-dev@continuum.io).

Source code for the latest development version of blaze can
be obtained [from Github](https://github.com/ContinuumIO/blaze).


Documentation
-------------

Documentation is available at
[blaze.pydata.org/docs/dev/](http://blaze.pydata.org/docs/dev/)


License
-------

Blaze development is sponsored by Continuum Analytics.

Released under BSD license. See [LICENSE.txt](LICENSE.txt) for details.
