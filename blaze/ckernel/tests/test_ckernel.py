import unittest

from blaze import ckernel

class TestCKernel(unittest.TestCase):
    def test_constructor_errors(self):
        self.assertRaises(TypeError, ckernel.CKernel, "wrong", "types")
        self.assertRaises(ValueError, ckernel.CKernel,
                        ckernel.DynamicKernelInstanceP(),
                        ckernel.UnarySingleOperation)

if __name__ == '__main__':
    unittest.main()

