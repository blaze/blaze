import unittest

from blaze import ckernel

class TestCKernel(unittest.TestCase):
    def test_constructor_errors(self):
        self.assertRaises(TypeError, ckernel.CKernel, "wrong", "types")

if __name__ == '__main__':
    unittest.main()

