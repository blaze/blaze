import sys
if len(sys.argv) > 1 and sys.argv[1] == 'develop':
    from setuptools import setup
else:
    from distutils.core import setup

setup(name='BlazeWeb',
      version='dev',
      description='Blaze Web-server/client',
      author='Continuum Analytics',
      author_email='mwiebe@continuum.io',
      url='http://www.continum.io/',
      packages=['blaze_web', 'blaze_web.common',
		'blaze_web.client', 'blaze_web.server'],
     )
