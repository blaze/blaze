{-# Language GADTs #-}
{-# Language RankNTypes #-}

-- Blaze DataShapes in Haskell

import Prelude hiding (Either, (.))
import Control.Monad

type Sym = String
newtype Env = Env [(Sym, Shape)] deriving Show

-- Types
data Type where
    Type  :: Shape -> Type
    Arrow :: Shape -> Shape -> Type
    deriving (Show, Read, Eq)

-- Shapes
data Shape where
    -- Atomic
    Fixed    :: Int -> Shape
    Range    :: Int -> Int -> Shape
    TypeVar  :: Sym -> Shape
    Bitfield :: Int -> Shape
    Bool     :: Bool -> Shape
    Ptr      :: Int -> AddrSpace -> Shape

    -- Composite
    Enum     :: [Shape] -> Shape
    Union    :: [Shape] -> Shape
    Either   :: Shape -> Shape -> Shape
    Record   :: [(String, Shape)] -> Shape
    Function :: Lambda -> Shape
    deriving (Show, Read, Eq)

-- Typed lambda expressions
data Lambda
    = Var Sym
    | App Lambda Lambda
    | Lam Sym Type Lambda
    deriving (Show, Read, Eq)

data AddrSpace = Local | Remote | Shared deriving (Show, Eq, Read)

class Typed x
class Shaped x
instance Typed Type
instance Shaped Shape

unify :: [Shape] -> [Shape] -> [Shape]
unify [] [] = []
unify (x:xs) (y:ys) = unify' x y : unify xs ys
    where
        unify' (Fixed a) (Fixed b) = case head xs of
            (Fixed 1) -> (Fixed a)
unify _ _ = error "size mismatch"


typeof :: Env -> Lambda -> Maybe Type
typeof (Env r) (Lam _ t _) = Just t
typeof (Env r) _ = undefined
{-typeof (Env r) (Var s) = lookup s r-}

lower :: Shape -> Int
lower (Range a _) = a
lower _ = undefined

upper :: Shape -> Int
upper (Range _ b) = b
upper _ = undefined

size :: Shape -> Int
size (Fixed n) = n
size (Bitfield n) = n
size _ = undefined

main :: IO Shape
main = forever $ do 
    x <- readLn :: IO Shape
    print $ show x
    return x
