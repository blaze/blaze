=======
Install
=======

Installing
~~~~~~~~~~

Blaze can be most easily installed from conda_

::

   $ conda install blaze

If you are interested in the development version of Blaze you can
obtain the source from Github.

::

    $ git clone git@github.com:ContinuumIO/blaze.git

However many of the dependencies are non-trivial to install.
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

    $ nosetests --with-doctest

Dependencies
~~~~~~~~~~~~

* numpy_ >= 1.6
* datashape_ >= 0.1.1
* dynd-python_ >= 0.6.1
* toolz_ >= 0.6.0
* multipledispatch_ >= 0.4.2
* unicodecsv_

**Optional**

* sqlalchemy_
* h5py_
* pandas_
* spark_ >= 1.0.0
* nose_ (for running tests)


.. _numpy: http://www.numpy.org/
.. _h5py: http://docs.h5py.org/en/latest/
.. _nose: https://pypi.python.org/pypi/nose/
.. _dynd-python: https://github.com/ContinuumIO/dynd-python
.. _datashape: https://github.com/ContinuumIO/datashape
.. _blz: https://github.com/ContinuumIO/blz
.. _spark: http://spark.apache.org/
.. _toolz: http://toolz.readthedocs.org/
.. _multipledispatch: http://multiple-dispatch.readthedocs.org/
.. _conda: http://conda.pydata.org/
.. _unicodecsv: https://github.com/jdunck/python-unicodecsv
