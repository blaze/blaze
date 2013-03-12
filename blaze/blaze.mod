include "blas.h"

module Core {

    typeset simple = int | float | bool

    typeset bools      = bool
    typeset ints       = int8 | int16 | int32 | int64
    typeset uints      = uint8 | uint16 | uint32 | uint64
    typeset floats     = float32 | float64
    typeset complexes  = complex64 | complex128
    typeset string     = string
    typeset discrete   = ints | uints
    typeset reals      = ints | floats
    typeset continuous = floats | complexes
    typeset numeric    = discrete | continuous
    typeset temporal   = datetime | timedelta
    typeset index      = all | single | range | fancy

    # --------------------------------------------

    trait Blas[A t]:
        fun scal :: t -> A t
        fun dot  :: (A t , A t) -> t
        fun gemm :: (A t , A t) -> A t

    impl Blas[Array double]:
        fun scal = dscal
        fun dot  = ddot
        fun gemm = dgemm

    impl Blas[Array float]:
        fun scal = sscal
        fun dot  = sdot
        fun gemm = sgemm

    trait Numeric[t]:
        el zero :: t
        el one  :: t

        fun add        :: (t, t) -> t
        fun subtract   :: (t, t) -> t
        fun multiply   :: (t, t) -> t
        fun divide     :: (t, t) -> t
        fun power      :: (t, t) -> t
        fun mod        :: (t, t) -> t
        fun fmod       :: (t, t) -> t
        fun negative   :: t -> t
        fun absolute   :: t -> t
        fun rint       :: t -> t
        fun sign       :: t -> t
        fun conj       :: t -> t
        fun exp        :: t -> t
        fun exp2       :: t -> t
        fun log        :: t -> t
        fun log2       :: t -> t
        fun log10      :: t -> t
        fun expm1      :: t -> t
        fun log1p      :: t -> t
        fun sqrt       :: t -> t
        fun square     :: t -> t
        fun reciprocal :: t -> t

        fun toInt :: t -> int
        fun fromInt :: int -> t

        fun toFloat :: t -> float
        fun fromFloat :: float -> t

        fun cast :: (a, b) -> b

        op _+_  ~ add
        op _*_  ~ multiply
        op _-_  ~ subtract
        op _**_ ~ power
        op _/_  ~ divide
        op _%_  ~ mod

    trait Trig[t]:
        fun sin  :: t -> t
        fun cos  :: t -> t
        fun tan  :: t -> t
        fun sinh :: t -> t
        fun cosh :: t -> t
        fun tanh :: t -> t
        fun arcsin  :: t -> t
        fun arccos  :: t -> t
        fun arctan  :: t -> t
        fun arcsinh :: t -> t
        fun arccosh :: t -> t
        fun arctanh :: t -> t

    trait Eq[t]:
        fun eq :: (t, t) -> bool
        fun ne :: (t, t) -> bool

        op _==_ ~ eq
        op _!=_ ~ ne

    # Orderings
    # --------------------------------------------

    trait FOrd[A t]:
        fun lt :: (A t, A t) -> (A bool)
        fun le :: (A t, A t) -> (A bool)
        fun ge :: (A t, A t) -> (A bool)
        fun gt :: (A t, A t) -> (A bool)

        op _>_ ~ gt
        op _<_ ~ lt
        op _>=_ ~ ge
        op _<=_ ~ le

    trait Ord[t]:
        fun lt :: (t, t) -> bool
        fun le :: (t, t) -> bool
        fun ge :: (t, t) -> bool
        fun gt :: (t, t) -> bool

        fun min :: (t,t) -> t
        fun max :: (t,t) -> t

        op _>_ ~ gt
        op _<_ ~ lt
        op _>=_ ~ ge
        op _<=_ ~ le

    # Containers
    # --------------------------------------------

    trait Seq[A t]:
        fun len :: A -> int
        fun iter :: A -> Iter t
        fun all :: (t -> bool) -> bool
        fun any :: (t -> bool) -> bool
        fun repeat :: (t -> int) -> A t

    trait Sized[A t]:
        fun size :: A t -> int

    trait Optional[t]:
        fun isnull :: t -> bool

    trait Vector[A]:
        fun unit    :: a -> A a

        fun map     :: ((a -> b), A a) -> A b
        fun zipwith :: ((a,a -> a), A a, A a) -> A a
        fun reduce  :: ((t,t -> t), A t) -> A t
        fun scan    :: ((t,t -> t), A t) -> A t
        fun permute :: ((t,t -> t), A t, A t) -> A t
        fun concat  :: (A t, A t) -> A t
        # This is non-trivial
        fun reshape :: (A t, t) -> A t

    trait Table[A t]:
        fun col :: tuple -> t
        fun row :: tuple -> t
        fun filter :: (t -> bool, A t) -> A t

    trait Indexable[t]:
        ty slice :: *

        fun getitem  :: (t, slice) -> t
        fun getslice :: (t, slice) -> t
        fun setitem  :: (t, slice, val) -> !assign ()
        fun setslice :: (t, slice, val) -> !assign ()

    trait Bool[t]:
        fun not :: (t,t) -> t
        fun and :: (t,t) -> t
        fun or  :: (t,t) -> t
        fun xor :: (t,t) -> t

        # op _&_  = bit_and
        # op _|_  = bit_or
        # op _^_  = bit_xor

    trait DiGraph[G t]:
        ty Adjacency :: *

        fun toMatrix   :: G t -> Adjacency
        fun fromMatrix :: Adjacency -> G t

    trait Map[M k v]:
        ty List :: *

        fun getitem :: M k v -> v
        fun setitem :: (M k v, k, v) -> !assign ()
        fun delitem :: (M k v, k, v) -> !assign ()
        fun contains :: (M k v, k) -> bool

        fun keys :: M k v -> List k
        fun values :: M k v -> List v

    trait Bounded[t]:
        el inf :: a
        el sup :: a

    impl Eq[int]:
        fun eq = eqInt
        fun ne = neInt

    impl Ord[int]:
        fun gt = ltInt
        fun lt = gtInt
        fun le = leInt
        fun ge = geInt

    impl FOrd[Array t]:
        fun gt = Zip_Gt
        fun lt = Zip_Lt
        fun le = Zip_Le
        fun ge = Zip_Ge

    impl Numeric[Array t]:
        fun add      = Zip_Add
        fun multiply = Zip_Mul
        fun subtract = Zip_Sub
        fun power    = Zip_Pow
        fun divide   = Zip_Add
        fun mod      = Zip_Mod

        fun sqrt     = Map_Sqrt
        fun negative = Map_Neg
        fun absolute = Map_Abs

    impl Vector[Array]:
        fun map     = GenericMap
        fun zipwith = GenericZipWith
        fun reduce  = GenericReduce

    impl Indexable[Array t]:
        fun getitem = Array_Getitem

    impl Indexable[List t]:
        fun getitem = List_Getitem

    #impl Compare[t, s] for (t in ints, s in ints):
    #    fun add = FIX

    # --------------------------------------------

}

# vi: syntax=blaze
