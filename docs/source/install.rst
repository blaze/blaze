=======
Install
=======

Installing
~~~~~~~~~~

If you are interested in the development version of Blaze you can
obtain the source from Github.

::

    $ git clone git@github.com:ContinuumIO/blaze-core.git

Many of the dependencies ( i.e. llvmpy ) are non-trivial to install.
It is **highly recommend** that you build Blaze using the Anaconda
distribution, a free Python distribution that comes with a host of
scientific and numeric packages bundled and precompiled into a userspace
Python environment.

Anaconda can be downloaded for all platforms here: http://continuum.io/anacondace.html .

Introduction
~~~~~~~~~~~~

To build project inside of Anaconda:

::

    $ make build

To build documentation:

::

    $ make docs

To run tests:

::

    $ python -m blaze.testing

Dependencies
~~~~~~~~~~~~

* numpy_ >= 1.5
* cython_ >= 0.11.1
* llvmpy_
* ply_
* python-blosc_
* pycparser_

.. _numpy: http://www.numpy.org/
.. _cython: http://www.cython.org/
.. _llvmpy: http://www.llvmpy.org/
.. _ply: http://www.dabeaz.com/ply/
.. _python-blosc: http://blosc.pytables.org
.. _pycparser: https://bitbucket.org/eliben/pycparser
