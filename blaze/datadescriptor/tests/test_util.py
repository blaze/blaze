from blaze.datadescriptor.util import *
from unittest import TestCase

class Test_openfile(TestCase):
    def test_openfile(self):
        with openfile() as f:
            with open(f, 'w') as a:
                a.write('')
            with openfile() as g:
                assert f != g

        assert not os.path.exists(f)
        assert not os.path.exists(f)
