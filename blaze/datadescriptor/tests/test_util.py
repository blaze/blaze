from blaze.datadescriptor.util import *
from unittest import TestCase

class Test_tmpfile(TestCase):
    def test_tmpfile(self):
        with tmpfile() as f:
            with open(f, 'w') as a:
                a.write('')
            with tmpfile() as g:
                assert f != g

        assert not os.path.exists(f)
        assert not os.path.exists(f)
