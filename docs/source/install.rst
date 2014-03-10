=======
Install
=======

Installing
~~~~~~~~~~

If you are interested in the development version of Blaze you can
obtain the source from Github.

::

    $ git clone git@github.com:ContinuumIO/blaze.git

Many of the dependencies ( i.e. llvmpy ) are non-trivial to install.
It is **highly recommend** that you build Blaze using the Anaconda
distribution, a free Python distribution that comes with a host of
scientific and numeric packages bundled and precompiled into a userspace
Python environment.

Anaconda can be downloaded for all platforms here:
http://continuum.io/anaconda.html .

Introduction
~~~~~~~~~~~~

To build project inside of Anaconda:

::

    $ python setup.py install

To build documentation on a unix-based system:

::

    $ cd docs
    $ make docs

To run tests:

::

    $ python -c 'import blaze; blaze.test()'

Dependencies
~~~~~~~~~~~~

* numpy_ >= 1.6
* llvmpy_>= 0.12
* pyparsing_ >= 2.0.1
* ply_ >= 3.4
* flask_ >= 0.10.1
* numba_ >= 0.11
* dynd-python_ >= 0.6.1
* datashape_ >= 0.1.1
* blz_ >= 0.6.1
* pykit_ >= 0.2.0
* nose_ (optional, for running tests)
* pytables_ >= 3.0.0 (optional, for hdf5 files)

.. _numpy: http://www.numpy.org/
.. _llvmpy: http://www.llvmpy.org/
.. _ply: http://www.dabeaz.com/ply/
.. _nose: https://pypi.python.org/pypi/nose/
.. _dynd-python: https://github.com/ContinuumIO/dynd-python
.. _datashape: https://github.com/ContinuumIO/datashape
.. _blz: https://github.com/ContinuumIO/blz
.. _pykit: https://github.com/pykit/pykit
.. _pytables: http://www.pytables.org/moin
.. _flask: http://flask.pocoo.org/
.. _numba: http://numba.pydata.org/
.. _pyparsing: http://pyparsing.wikispaces.com/
