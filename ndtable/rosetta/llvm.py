# --------------------------------
# Rosetta stone of LLVM and Blaze
# --------------------------------

Type.int() is int32
Type.int(8) is int8
Type.int(16) is int16
Type.int(32) is int32

Type.double() is float64

Type.fp128() is float128

Type.array(T, x) is (x, T)

Type.function(ret, arg) is (ret -> arg)

Type.pointer(ptr) is *ptr
Type.pointer(ptr, addr) is *(ptr, addr)

Type.struct is Record
