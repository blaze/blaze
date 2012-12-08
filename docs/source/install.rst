=======
Install
=======

Installing
~~~~~~~~~~

If you are interested in the development version of Blaze you can
obtain the source from Github.

::

    $ git clone git@github.com:ContinuumIO/blaze.git

Many of the dependencies ( llvm, numba, ... ) are non-trivial to
install. It is highly recommend that you build Blaze using the Anaconda
Python distribution.

Free Anaconda CE is available here: http://continuum.io/anacondace.html .

Dependencies
~~~~~~~~~~~~

You will need to have the `ATerm library <http://strategoxt.org/Tools/ATermLibrary>`_ installed in order to build Blaze.  Installation instructions:

::

    Debian     : apt-get install libaterm
    Arch Linux : pacman -S libaterm
    Mac        : brew install aterm   or  port install libaterm
    Windows    : ftp://ftp.stratego-language.org/pub/stratego/StrategoXT/strategoxt-0.17/cygwin/aterm-2.5-cygwin.tar.gz
    Other      : ftp://ftp.stratego-language.org/pub/stratego/StrategoXT/strategoxt-0.17/aterm-2.5.tar.gz 

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
