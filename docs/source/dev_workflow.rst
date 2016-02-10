==================
Developer Workflow
==================

This page describes how to install and improve the development version of Blaze.

If this documentation isn't sufficiently clear or if you have other questions
then please email blaze-dev@continuum.io.

Installing Development Blaze
----------------------------

Blaze depends on many other projects, both projects that develop alongside
blaze (like ``odo``) as well a number of community projects (like ``pandas``).

Blaze development happens in the following projects, all of which are available
on github.com/blaze/project-name

*  Blaze_
*  DataShape_
*  Odo_
*  Dask_

Bleeding edge binaries are kept up-to-date on the ``blaze`` conda channel.
New developers likely only need to interact with one or two of these libraries so we recommend downloading everything by the conda channel and then only cloning those git repositories that you actively need::

    conda install -c blaze blaze  # install everything from dev channel
    git clone git://github.com/blaze/blaze.git  # only clone blaze and odo
    git clone git://github.com/blaze/odo.git  # only clone blaze and odo

.. _Odo: https://github.com/blaze/odo
.. _Dask: https://github.com/blaze/dask
.. _Blaze: https://github.com/blaze/blaze
.. _DataShape: https://github.com/blaze/datashape
.. _conda: http://conda.pydata.org/
.. _Anaconda: http://continuum.io/downloads
.. _anaconda.org: https://anaconda.org/


GitHub Flow
-----------

Source code and issue management are hosted in `this github page`_,
and usage of git roughly follows `GitHub Flow`_. What this means
is that the `master` branch is generally expected to be stable,
with tests passing on all platforms, and features are developed in
descriptively named feature branches and merged via github's
Pull Requests.

.. _this github page: https://github.com/blaze/blaze
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
OS X, and Windows.   Further tests and bleeding-edge builds are carried out
using `Anaconda build` which tests and builds Blaze on the following
platforms/versions

*   Python versions 2.6, 2.7, 3.3, 3.4
*   Operating systems Windows, OS-X, Linux
*   32-bit and 64-bit

.. _`Anaconda build`: https://anaconda.org/blaze/blaze/builds


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
