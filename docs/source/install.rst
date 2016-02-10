=======
Install
=======

Installing
~~~~~~~~~~

Blaze can be most easily installed from conda_

::

   $ conda install blaze

More up-to-date builds are available on the ``blaze`` anaconda channel:
http://anaconda.org/blaze

::

    conda install -c blaze blaze

Blaze may also be installed using ``pip``:

::

    pip install blaze --upgrade
    or
    pip install git+https://github.com/blaze/blaze  --upgrade

If you are interested in the development version of Blaze you can
obtain the source from Github.

::

    $ git clone git@github.com:blaze/blaze.git

Anaconda can be downloaded for all platforms here:
http://continuum.io/anaconda.html .

Introduction
~~~~~~~~~~~~

To build project from source:

::

    $ python setup.py install

To build documentation on a unix-based system:

::

    $ cd docs
    $ make docs

To run tests:

::

    $ py.test --doctest-modules --pyargs blaze

Strict Dependencies
~~~~~~~~~~~~~~~~~~~

Blaze depends on NumPy, Pandas, and a few pure-python libraries.  It should be
easy to install on any Numeric Python setup.

* numpy_ >= 1.7
* datashape_ >= 0.4.4
* odo_ >= 0.3.1
* toolz_ >= 0.7.0
* cytoolz_
* multipledispatch_ >= 0.4.7
* pandas_

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Blaze can help you use a variety of other libraries like ``sqlalchemy`` or
``h5py``.  If these are installed then Blaze will use them.  Some of these are
non-trivial to install.  We recommend installation throgh ``conda``.

* sqlalchemy_
* h5py_
* spark_ >= 1.1.0
* pymongo_
* pytables_
* bcolz_
* flask_ >= 0.10.1
* pytest_ (for running tests)


.. _numpy: http://www.numpy.org/
.. _odo: https://github.com/blaze/odo
.. _h5py: http://docs.h5py.org/en/latest/
.. _pytest: http://pytest.org/latest/
.. _datashape: https://github.com/blaze/datashape
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
