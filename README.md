Blaze
=====

![](https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/numpy_plus.png)

Blaze is the next-generation of NumPy. It is designed as a foundational
set of abstractions on which to build out-of-core and distributed
algorithms over a wide variety of data sources.

![](https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/sources.png)

Blaze is a work in progress at the moment. The code is quite a distance
from feature complete. The code is released in an effort to start a
public discussion with our end users and community.

Documentation
-------------

* [0.1 Dev Docs](http://blaze.pydata.org/docs/)

Installing
----------

If you are interested in the development version of Blaze you can
obtain the source from Github.

```bash
$ git clone git@github.com:ContinuumIO/blaze.git
```

Many of the dependencies ( llvm, numba, ... ) are non-trivial to
install. It is highly recommend that you build Blaze using the Anaconda
Python distribution.

Free Anaconda CE is available here: http://continuum.io/anacondace.html .

Using Anaconda's package manager:

```bash
$ conda install ply
$ conda install blosc
```

Introduction
------------

To build project inside of Anaconda:

```bash
$ make build
```

To build documentation:

```bash
$ make docs
```

To run tests:

```bash
$ python setup.py test
```

Contributing
------------

Anyone wishing to contribute should join the discussion on the
[blaze-dev](https://groups.google.com/a/continuum.io/forum/#!forum/blaze-dev)
mailing list at: blaze-dev@continuum.io

License
-------

Blaze development is sponsored by Continuum Analytics.

Released under BSD license. See LICENSE for details.
