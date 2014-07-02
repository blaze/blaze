<p align="center" style="padding: 20px">
<img src="https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/blaze_med.png">
</p>

**Blaze** extends the usability of NumPy and Pandas to distributed and
out-of-core computing.  Blaze provides an interface similar to that of the
NumPy ND-Array or Pandas DataFrame.  Blaze maps inputs from these familiar
interfaces onto a variety of other computational engines like Postgres or
Spark.  Blaze connects users to big computation from the comfort of an
interactive object.

<p align="center" style="padding: 20px">
<img src="https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/numpy_plus.png">
</p>

Blaze includes a rich set of computational and data primitives useful in
building and communicating between computational systems.  Blaze can help with
consistent and robust data migration, as well as remote execution.

<p align="center" style="padding: 20px">
<img src="https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/codepush.png">
</p>

Continuum Analytics' vision is to provide open technologies for data
integration on a massive scale based on a vision of a structured,
universal "data web". In the same way that URL, HTML, and HTTP form
the basis of the World Wide Web for documents, Blaze could
be a fabric for structured and numerical data spearheading
innovations in data management, analytics, and distributed computation.

Blaze aims to be a foundational project allowing many different users of
other PyData projects (Pandas, Theano, Numba, SciPy, Scikit-Learn)
to interoperate at the application level and at the library level with
the goal of being able to to lift their existing functionality into a
distributed context.

<p align="center" style="padding: 20px">
<img src="https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/sources.png">
</p>

Status
------

Blaze is in development.
Take a look at the [release notes](docs/source/releases.rst).

Documentation
-------------

Documentation is available at [blaze.pydata.org/](http://blaze.pydata.org/)

Trying out Blaze
----------------

The easiest way to try out Blaze is through the
[Anaconda distribution](http://continuum.io/downloads).

To make sure you're running the latest released version
of Blaze, use the
[conda package manager](http://docs.continuum.io/conda/index.html)
to update.

```bash
$ conda update blaze
```

Source code for the latest development version of blaze can
be obtained [from Github](https://github.com/ContinuumIO/blaze).

Dependencies
------------

The Blaze project itself is spread out over multiple projects,
in addition to the main `blaze` repo. Other Blaze projects include the
following:

  * [datashape][datashape]
  * [dynd-python][dynd-python]

Additionally, Blaze hooks into and depends on several excellent projects
written by the general community.  See `requirements.txt` for a full list of
dependencies.

[datashape]: https://github.com/ContinuumIO/datashape
[dynd-python]: https://github.com/ContinuumIO/dynd-python


Installing from Source
----------------------

Install all the pre-requisites using conda or another mechanism,
then run:

```bash
$ python setup.py install
```

Installing from Binstar.org
---------------------------

If you're on a Linux or Mac OS-X platform, you can install a development
version of Blaze (hosted on Binstar) by typing the following:

```bash
$ conda install -c mwiebe -c mrocklin blaze
```

Contributing
------------

Anyone wishing to discuss on Blaze should join the
[blaze-dev](https://groups.google.com/a/continuum.io/forum/#!forum/blaze-dev)
mailing list. To get started contributing, read through the
[Developer Workflow](docs/source/dev_workflow.rst) documentation.

License
-------

Blaze development is sponsored by Continuum Analytics.

Released under BSD license. See [LICENSE.txt](LICENSE.txt) for details.
