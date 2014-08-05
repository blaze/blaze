==================
Developer Workflow
==================

This page describes how to install and improve the development version of Blaze.

If this documentation isn't sufficiently clear or if you have other questions
then please email blaze-dev@continuum.io.

Installing Development Blaze
----------------------------

Blaze has a number of dependencies, both due to relying on community projects
and also because it itself is split among a few projects.

**TL;DR**:  Install `conda_`, then copy-paste the following ::

   # Get source
   git clone https://github.com/ContinuumIO/blaze.git
   cd blaze

   # Install most of the requirements through conda
   conda install --yes --file requirements.txt  # Basic requirements
   conda install --yes -c mwiebe dynd-python    # Development version of DyND
   conda install --yes unicodecsv               # If on Python 2

   # Some requirements are kept up-to-date on PyPI
   conda install --yes pip
   pip install git+https://github.com/ContinuumIO/datashape.git --upgrade
   pip install toolz cytoolz multipledispatch  --upgrade

   # Install and run tests
   python setup.py install                      # Install Blaze
   python -c 'import blaze; blaze.test()'       # Run tests


**Detailed Version**:

Get Source:  Fork and clone the Blaze git repository to your local machine (see
Github Flow below)

::

   git clone git@github.com:ContinuumIO/blaze.git
   or
   git clone git@github.com:GITHUB-USERNAME/blaze.git

Install dependencies:  Blaze depends on a number of projects within the larger
scientific Python ecosystem.  We strongly suggest that you install these
dependencies using `conda`_, preferably from the `Anaconda`_ download.

::

   conda install --file requirements.txt

It may be that newer versions of these projects have critical updates that have
not yet reached the main conda channel.  Because of this we recommend
installing with some additional Binstar channels.

::

   conda install -c mwiebe -c mrocklin --file requirements.txt

If you don't want to use conda that's fine too.  Most of the requirements can
be installed from PyPI.  A notable exception is `DyND`_ which must be compiled
from source.

If you use Python 2 you should also include the ``unicodecsv`` module

::

   conda install unicodecsv

.. _DyND: https://github.com/ContinuumIO/dynd-python
.. _conda: http://conda.pydata.org/
.. _Anaconda: http://continuum.io/downloads
.. _binstar: https://binstar.org/


GitHub Flow
-----------

Source code and issue management are hosted in `this github page`_,
and usage of git roughly follows `GitHub Flow`_. What this means
is that the `master` branch is generally expected to be stable,
with tests passing on all platforms, and features are developed in
descriptively named feature branches and merged via github's
Pull Requests.

.. _this github page: https://github.com/ContinuumIO/blaze
.. _GitHub Flow: http://scottchacon.com/2011/08/31/github-flow.html


Coding Standards
----------------

**Unified Python 2 and 3 Codebase:**

Blaze source code simultaneously supports both Python 2 and Python 3 with a
single codebase.

To support this, all .py files must begin with a few `__future__`
imports, as follows.::

    from __future__ import absolute_import, division, print_function


**Testing:**

In order to keep the ``master`` branch functioning with passing tests,
there are two automated testing mechanisms being used. First is
`Travis CI`_, which is configured to automatically build any pull
requests that are made. This provides a smoke test against both
Python 2 and Python 3 before a merge.

.. _Travis CI: https://travis-ci.org/

The Travis tests only run on Linux, but Blaze is supported on Linux,
OS X, and Windows. Internal to Continuum, a `Jenkins`_ server is
running which builds and tests Blaze on 15 different configurations,
versions 2.6, 2.7, and 3.3 for different 32-bit and 64-bit versions
of Linux, OS X, and Windows. That these configurations are all working
should be verified by someone at Continuum after each merge of a
pull request.

.. _Jenkins: http://jenkins-ci.org/


**Relative Imports:**

To avoid the side effects of top level imports, e.g. `import blaze`, all internal code should be imported relatively.  Thus::

    #file: blaze/objects/table.py
    from blaze import Array

should be::

     #file: blaze/objects/table.py
     from .array import Array

For cross submodule imports, import from the module api.  For example::

    #file: blaze/objects/table.py
    from ..io import printing

Relation with Continuum
-----------------------

Blaze is developed in part by `Continuum Analytics`_, a for profit company.
Continuum's efforts on Blaze are open source and freely available to the public.
The open nature of Blaze is protected by a BSD license.

.. _Continuum Analytics: http://continuum.io/
