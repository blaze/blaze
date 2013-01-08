# Array indexing
interface Ix t:
    fun getitem  :: (t, index) -> t
    fun getslice :: (t, index) -> t

    fun setitem  :: (t, index val) -> ()
    fun setslice :: (t, index val) -> ()

# Scalar arithmetic
interface Arith a b:
    op _+_ :: (a,a) -> a
    op _-_ :: (a,a) -> a
    op _*_ :: (a,a) -> a

    op _+_ :: (b,b) -> b
    op _-_ :: (b,b) -> b
    op _*_ :: (b,b) -> b

    op _**_ :: (a,a) -> a
    op _**_ :: (a,b) -> b
    op _**_ :: (b,a) -> b

    fun abs :: a -> a
    fun abs :: b -> b

    fun exp :: b -> b
    fun log :: b -> b
    fun sqrt :: b -> b

    fun sin :: b -> b
    fun cos :: b -> b
    fun tan :: b -> b

    fun asin :: b -> b
    fun acos :: b -> b
    fun atan :: b -> b

    fun asinh :: b -> b
    fun acosh :: b -> b
    fun atanh :: b -> b

interface Ord t:
    op _>_ :: (t,t) -> t
    op _<_ :: (t,t) -> t

interface Bool t:
    fun or :: (t, t) -> t
    fun and :: (t, t) -> t
    fun xor :: (t, t) -> t

interface Traversable t:
    fun map :: (f, t) -> t
    fun zip :: (f, t) -> t
