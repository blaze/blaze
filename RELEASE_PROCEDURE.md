Blaze Release Procedure
=======================

This document describes the steps to follow to release
a new version of Blaze.

1. Update version numbers in the following locations:

 * /setup.py
 * /blaze/__init__.py
 * /conda.yaml

2. Confirm the dependencies and their version numbers in
   /docs/source/install.rst
   /requirements.txt
   In particular, `datashape`, `dynd-python`, etc
   will typically be released concurrently with `blaze`,
   so they need to be updated to match.

3. Update the release notes /docs/source/releases.rst
   You may use a github URL like https://github.com/ContinuumIO/blaze/compare/0.6.0...master for assistance.

4. Build and update the documentation in gh-pages.

5. Verify build is working on all platforms. The
   jenkins builder internal to Continuum can assist
   with this.

6. Tag the release version.

7. Release email to blaze-dev@continuum.io.

8. Update this release procedure document to reflect
   what needed to be done for the release.
