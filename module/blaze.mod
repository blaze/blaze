# Array objects
interface Array A a:
    fun map        :: ((a -> b), A a) -> A b
    fun reduce     :: (((a,a) -> b), A a) -> A b
    fun accumulate :: (((a,a) -> b), A a) -> A b
    fun zipwith    :: (((a,b) -> c), A a, A b) -> A c

    #op _+_ = (A a, A a) -> zipwith(<add>, A a, A a)
    #op _+_ = (A a, A b) -> zipwith(<add>, A c, A c) where
    #    { a', b' = <unify(A a, A b)> }


# Array indexing
interface Ix T a:
    fun getitem  :: (T, index) -> T
    fun getslice :: (T, index) -> T

    fun setitem  :: (T, index, a) -> ()
    fun setslice :: (T, index, a) -> ()

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

    fun arcsin :: b -> b
    fun arccos :: b -> b
    fun arctan :: b -> b

    fun arcsinh :: b -> b
    fun arccosh :: b -> b
    fun arctanh :: b -> b


interface Ord T:
    op _>_ :: (T,T) -> T
    op _<_ :: (T,T) -> T


interface Bool T:
    fun or  :: (T, T) -> T
    fun and :: (T, T) -> T
    fun xor :: (T, T) -> T
