interface Array T ds:
    op _+_ :: (a,a) -> a

# Array indexing
interface Ix T:
    fun getitem  :: (T, index) -> T
    fun getslice :: (T, index) -> T

    fun setitem  :: (T, index, val) -> ()
    fun setslice :: (T, index, val) -> ()

    fun delitem  :: (T, index) -> ()
    fun delslice :: (T, index) -> ()


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
    op _**_ :: (b,b) -> b

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

interface Ord T:
    op _>_ :: (T,T) -> T
    op _<_ :: (T,T) -> T

interface Bool t:
    fun or  :: (T, T) -> T
    fun and :: (T, T) -> T
    fun xor :: (T, T) -> T

interface Traversable t:
    fun map :: ((a->b), T) -> T
    fun zip :: ((a->b), T) -> T
