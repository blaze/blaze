Blaze Release Procedure
=======================

1. Run the conda build recipes across all platforms (Linux, OS X,
   Windows) including both 32 and 64-bit architectures.

2. Tag the release version.

```sh
git tag -a x.x.x -m 'Version x.x.x'
```

3. Push those tags to `blaze` master

```sh
git push --tags upstream master
```

3. Upload a tarball to PyPI

```sh
python setup.py sdist upload  # from within your clone of blaze
```

4. Release email to blaze-dev@continuum.io.
