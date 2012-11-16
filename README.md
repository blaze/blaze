Blaze
=====

Blaze is the next-generation of NumPy. It is designed as a foundational
set of abstractions on which to build out-of-core and distributed
algorithms over a wide variety of data sources.

Blaze is a work in progress at the moment. The code is quite a distance
from feature complete. The code is released in an effort to start a
public discussion with our end users and community.

Blaze is approaching being a usable library, but does not have runnable
components at the moment.

Installing
----------

If you are interested in the development version of Blaze you can
obtain the source from Github.

```bash
$ git clone git@github.com:ContinuumIO/blaze.git
```

All dependencies will be resolved by working in the Anaconda environment.

Free Anaconda CE is available here: http://continuum.io/downloads.html.

Introduction
------------

To build project inside of Anaconda:

```bash
$ git submodule init
$ git submodule update
$ python setup.py install
```

To build documentation:

```bash
$ cd docs
$ make html
```

To run tests:

```bash
$ python setup.py test
```

Contributing
------------

Anyone wishing to contribute should join the discussion on the mailing
list at: blaze-dev@continuum.io

License
-------

Blaze development is sponsored by Continuum Analytics.

Released under BSD license. See LICENSE
