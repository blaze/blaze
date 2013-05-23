import llvm.core as lc
from .btypes import int_type, float_type

#------------------------------------------------------------------------
# Intrinsics Signatures
#------------------------------------------------------------------------

# trigonometric
sin  = 'sin', float_type, [float_type]
cos  = 'cos', float_type, [float_type]
tan  = 'tan', float_type, [float_type]
asin = 'asin', float_type, [float_type]
acos = 'acos', float_type, [float_type]
atan = 'atan', float_type, [float_type]

# hyperbolic
sinh = 'sinh', float_type, [float_type]
cosh = 'cosh', float_type, [float_type]
tanh = 'tanh', float_type, [float_type]

# power
sqrt = 'sqrt', float_type, [float_type]
pow  = 'pow', float_type, [float_type, float_type]

# exponential
exp   = 'exp', float_type, [float_type]
log   = 'log', float_type, [float_type]
log10 = 'log10', float_type, [float_type]

# misc
abs   = 'abs', float_type, [float_type]
ceil  = 'ceil', float_type, [float_type]
floor = 'floor', float_type, [float_type]
fmod  = 'fmod', float_type, [float_type, float_type]

# byteswapping

bswap = 'bswap', int_type, [int_type]

#------------------------------------------------------------------------
# Intrinsic Symbols
#------------------------------------------------------------------------

llvm_intrinsics = {
    'sin'   : lc.INTR_SIN,
    'cos'   : lc.INTR_COS,

    'sqrt'  : lc.INTR_SQRT,
    'pow'   : lc.INTR_POW,
    'exp'   : lc.INTR_EXP,
    'log'   : lc.INTR_LOG,
    'sqrt'  : lc.INTR_SQRT,
    'abs'   : lc.INTR_FABS,

    'bswap' : lc.INTR_BSWAP,
}
