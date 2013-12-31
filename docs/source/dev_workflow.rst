==================
Developer Workflow 
==================

Blaze is being developed as an open source project by
`Continuum Analytics`_. This page describes the basic workflow
that development follows, and may be used as a guide for how
to contribute to the project.

.. _Continuum Analytics: http://continuum.io/

GitHub Flow
~~~~~~~~~~~

Source code and issue management are hosted in `this github page`_,
and usage of git roughly follows `GitHub Flow`_. What this means
is that the `master` branch is generally expected to be stable,
with tests passing on all platforms, and features are developed in
descriptively named feature branches and merged via github's
Pull Requests.

.. _this github page: https://github.com/ContinuumIO/blaze
.. _GitHub Flow: http://scottchacon.com/2011/08/31/github-flow.html

Unified Python 2 and 3 Codebase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Blaze is being developed with one codebase that simultaneously
supports both Python 2 and Python 3. To make this manageable,
just Python 2.6, 2.7, and >= 3.3 are being supported.

To support this, all .py files must begin with a few `__future__`
imports, as follows.::

    from __future__ import absolute_import, division, print_function

Additionally, there is a helper submodule `blaze.py2help` including
a number of utilities to smooth the differences between Python versions.
Please browse the source file to see what is there, and read the
documentation for the `six library`_ for details on this style of
unified codebase.

.. _six library: http://pythonhosted.org/six/

Testing
~~~~~~~


