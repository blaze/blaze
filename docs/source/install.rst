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

    $ make docs

To run tests:

::

    $ python -c 'import blaze; blaze.test()'

Dependencies
~~~~~~~~~~~~

* numpy_ >= 1.5
* cython_ >= 0.11.1
* llvmpy_
* ply_
* python-blosc_
* pycparser_
* dynd_ (master branch)
* nose_ (optional, for running tests)

.. _numpy: http://www.numpy.org/
.. _cython: http://www.cython.org/
.. _llvmpy: http://www.llvmpy.org/
.. _ply: http://www.dabeaz.com/ply/
.. _python-blosc: http://blosc.pytables.org
.. _pycparser: https://bitbucket.org/eliben/pycparser
.. _nose: https://pypi.python.org/pypi/nose/
.. _dynd: https://github.com/ContinuumIO/dynd-python

