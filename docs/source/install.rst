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
