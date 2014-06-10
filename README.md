<p align="center" style="padding: 20px">
<img src="https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/blaze_med.png">
</p>

**Blaze** is the next-generation of NumPy. It is designed as a
foundational set of abstractions on which to build out-of-core and
distributed algorithms over a wide variety of data sources and to extend
the structure of NumPy itself.

<p align="center" style="padding: 20px">
<img src="https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/numpy_plus.png">
</p>

Blaze allows easy composition of low level computation kernels
( C, Fortran, Numba ) to form complex data transformations on large
datasets.

In Blaze, computations are described in a high-level language
(Python) but executed on a low-level runtime (outside of Python),
enabling the easy mapping of high-level expertise to data without sacrificing
low-level performance. Blaze aims to bring Python and NumPy into the
massively-multicore arena, allowing it to able to leverage many CPU and
GPU cores across computers, virtual machines and cloud services.

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

Blaze is a work in progress at the moment, currently at release 0.4.1-dev.
Take a look at the [release notes](docs/source/releases.rst).

Documentation
-------------

* [Dev Docs](http://blaze.pydata.org/docs/)

Trying out Blaze
----------------

The easiest way to try out Blaze is through the Anaconda
distribution. The latest release includes a version of Blaze.

http://continuum.io/downloads

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
in addition to the main `blaze` repo. These dependencies
are

  * [blz][blz]
  * [datashape][datashape]
  * [dynd-python][dynd-python]
  * [pykit][pykit]

To see a full list of dependencies, please see `requirements.txt`

[blz]: https://github.com/ContinuumIO/blz
[datashape]: https://github.com/ContinuumIO/datashape
[dynd-python]: https://github.com/ContinuumIO/dynd-python
[pykit]: https://github.com/pykit/pykit

Installing from Source
----------------------

Install all the pre-requisites using conda or another mechanism,
then run:

```bash
$ python setup.py install
```

Installing from Binstar.org
----------------------

Binstar is a service that allows you to create and manage public
and private package registries. 
If you're on a Linux or Mac OS X platform, you can install a binary
version of blaze (hosted on binstar) by typing:

```bash
$ conda install -c mwiebe -c mrocklin blaze
```


Documentation is generated using sphinx from the docs directory.

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
