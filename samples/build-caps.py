# Example that shows how to build a Blaze array with different
# capabilities

import blaze

# A default array (backed by NumPy)
a = blaze.array([1,2,3])
print a
print a[0]

# A compressed array (backed by BLZ)
b = blaze.array([1,2,3], caps={'compress': True})
print b
print b[0]

