=======
Install
=======

Installing
~~~~~~~~~~

If you are interested in the development version of Blaze you can
obtain the source from Github.

::

    $ git clone git@github.com:ContinuumIO/blaze-core.git

Many of the dependencies ( llvm, numba, ... ) are non-trivial to
install. It is highly recommend that you build Blaze using the Anaconda
Python distribution.

Free Anaconda CE is available here: http://continuum.io/anacondace.html .

Using Anaconda's package manager:

::

    $ conda install ply
    $ conda install blosc

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

    $ python setup.py test
