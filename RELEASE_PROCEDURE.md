Blaze Release Procedure
=======================

This document describes the steps to follow to release
a new version of Blaze.

1. Update version numbers in the following locations:

 * /setup.py
 * /blaze/__init__.py

2. Confirm the dependencies and their version numbers in
   /docs/source/install.rst
   /requirements.txt
   In particular, `datashape`, `odo`, etc
   will typically be released concurrently with `blaze`,
   so they need to be updated to match.

3. Update the release notes /docs/source/releases.rst
   You may use a github URL like https://github.com/ContinuumIO/blaze/compare/0.6.0...master for assistance.

4. Verify build is working on all platforms.  Binstar-build
   can assist with this.

5. Tag the release version.

        git tag -a x.x.x -m 'Version x.x.x'

    And push those tags

        git push --tags

6. Release email to blaze-dev@continuum.io.

7. Update this release procedure document to reflect
   what needed to be done for the release.
