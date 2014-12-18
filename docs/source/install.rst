=======
Install
=======

Installing
~~~~~~~~~~

Blaze can be most easily installed from conda_

::

   $ conda install blaze

More up-to-date builds are available on the ``blaze`` binstar channel:
http://binstar.org/blaze

::

    conda install -c blaze blaze

Blaze may also be installed using ``pip``:

::

    pip install blaze

If you are interested in the development version of Blaze you can
obtain the source from Github.

::

    $ git clone git@github.com:ContinuumIO/blaze.git

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

    $ py.test --doctest-modules --pyargs blaze

Dependencies
~~~~~~~~~~~~

* numpy_ >= 1.7
* datashape_ >= 0.4.0
* into_ >= 0.1.3
* dynd-python_ >= 0.6.4
* toolz_ >= 0.7.0
* cytoolz_
* multipledispatch_ >= 0.4.7
* pandas_

**Optional**

* dynd-python_ >= 0.6.5
* sqlalchemy_
* h5py_
* spark_ >= 1.0.0
* pymongo_
* pytables_
* bcolz_
* flask_ >= 0.10.1
* pytest_ (for running tests)


.. _numpy: http://www.numpy.org/
.. _into: https://github.com/ContinuumIO/into
.. _h5py: http://docs.h5py.org/en/latest/
.. _pytest: http://pytest.org/latest/
.. _dynd-python: https://github.com/ContinuumIO/dynd-python
.. _datashape: https://github.com/ContinuumIO/datashape
.. _pandas: http://pandas.pydata.org/
.. _cytoolz: https://github.com/pytoolz/cytoolz/
.. _sqlalchemy: http://www.sqlalchemy.org/
.. _spark: http://spark.apache.org/
.. _toolz: http://toolz.readthedocs.org/
.. _multipledispatch: http://multiple-dispatch.readthedocs.org/
.. _conda: http://conda.pydata.org/
.. _pymongo: http://api.mongodb.org/python/current/
.. _pytables: http://www.pytables.org/moin
.. _bcolz: https://github.com/Blosc/bcolz
.. _flask: http://flask.pocoo.org/
