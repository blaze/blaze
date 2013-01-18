=====
ATerm
=====

The Blaze python implementation of ATerm is designed to bring some of
the conveniences of pattern matching and rewriting capabilities of
languages like ML and Haskell into the Python world.

The ATerm representation of expression graphs forms the flexible
intermediate form that Blaze expressions can be manipulated with
in order to do optimization passes and execution dispatch.

Grammar
-------

The ATerm spec is overseen by

.. _StrategoXT Project: http://strategoxt.org/Tools/ATermLibrary

The ``blaze.aterm`` is conforming implementation of the specification
with the exception that we exclude Binary Large Objects ``blob`` objects
are not present due to limitations in Python and limited usefullness in
Blaze.

The abstract grammar for ATerm is shown below.

::

    t : bt                 -- basic term
      | bt {ty,m1,...}     -- annotated term

    bt : C                 -- constant
       | C(t1,...,tn)      -- n-ary constructor
       | (t1,...,tn)       -- n-ary tuple
       | [t1,...,tn]       -- list
       | "ccc"             -- quoted string ( explicit double quotes )
       | int               -- integer
       | real              -- floating point number

Types
-----

Example of ATerm expressions by category

:Integer: Integer terms (``1``)
:Real: Real terms (``2.718``)
:String: String terms (``"foo"``)
:Constants: Variable terms (``x``)
:Application: Application of constant terms to other subterms (``f(x,y)``)
:List: A ordered list of subterms (``[1,2,3]``)
:Tuple: A ordered tuple of subterms (``(1,2,3)``)
:Placeholder: A placeholder for use in ``match`` and ``make`` commands (``<int>``)

Constants
~~~~~~~~~

Constant symbols are simply letters. These are completely
context-free, they have no meaning in and of themselves.

.. code-block:: text

    x

.. code-block:: text

    foo

Integers
~~~~~~~~

.. code-block:: text

    1

Reals
~~~~~

.. code-block:: text

    3.14159

Strings
~~~~~~~

.. code-block:: text

    "foo"

.. code-block:: text

    "foo \"bar\" fizz"

Strings must use double quotes, and can escape inner quotes using
backslash.

Application
~~~~~~~~~~~

.. code-block:: text

    f(x,y)

.. code-block:: text

    f(1,g(2,3))

.. code-block:: text

    Add(1,2)

In the above example the operand being applied is referred to as
the ``spine`` of the application with ``arguments``.

Annotations
~~~~~~~~~~~

.. code-block:: text

    x{prop}

.. code-block:: text

    x{prop1, prop2}

.. code-block:: text

    f(x{prop1}, y{prop2})

Annotations can contain any number of comma seperated aterm expression
with the restriction that annotations cannot themselves be annotated.

Lists
~~~~~

.. code-block:: text

   [1,2,3]

.. code-block:: text

   [3.141, 1, foo]

**Tuples**

.. code-block:: text

   (1, 2.718, 3.141)

.. code-block:: text

   (foo, bar)


Usage
-----

The aterm library provides three common operations that are used
for construction and deconstruction aterm expressions. 

::

    from blaze.aterm import parse, match, build

:parse: Maps strings into ATerm expressions.
:match: Deconstructs ATerm expressions using pattern matching.
:build: Constructs ATerm expressions using pattern matching.


::

    >>> parse('x')
    x

::

    >>> parse('f(x,y)')
    f(x,y)

::

    >>> parse('x{prop1}')
    x{(prop1,)}


Pattern Matching
----------------

Pattern matching is the action of determining whether a given aterm
expression conforms to pattern, similar in notion to regex. The pattern
may also contain placeholder objects which can be used to deconstruct
and pluck values of out aterms while rewriting.


:`<int>`: Matches int terms
:`<real>`: Matches real number terms
:`<str>`: Matches str terms
:`<term>`: Matches all terms
:`<placeholder>`: Matches all placeholder terms
:`<appl(...)>`: Matches all application to the specific arguments, ``(...)`` is not
    aterm syntax. See examples below.

The result a pattern match is a 2-tuple containing a boolean
indicating whether the match succeeded and a list containing the
capture valued.


::

    >>> match('x', 'x')
    (True, [])

::

    >>> match('x', 'y')
    (False, [])

::

    >>> match('f(<int>,<int>)', 'f(1,2)')
    (True, [1,2])

::

    >>> match('<term>', 'x')
    (True, [x])

::

    >>> match('f(<real>)', 'f(1)')
    (False, [])

::

    >>> match('Add(Succ(<int>), <term>)', 'Add(Succ(2), Succ(3))')
    (True, [2, Succ(3)]

::

    >>> match('<appl(x,3)>', 'f(x,3)')
    (True, [f])


For those coming from other languages, an analogy is useful. The
match operator in Prolog is written with `?`.

.. code-block:: prolog

    ?- x = x.
    true
    ?- x = y.
    false

Or often used to define functions which operate over pattern
matched variables collected on the LHS to free variables on the
RHS. For example in Prolog:

.. code-block:: prolog

    fact(0) => 0
    fact(n) => n*fact(n-1);

Or in ML:

.. code-block:: ocaml

    fun fact(1) = 1
      | fact(n) = n*fact(n-1);

Or in Haskell:

.. code-block:: haskell

    fact 0 = 1
    fact n = n * fact (n-1)

Pretty Printing
~~~~~~~~~~~~~~~

The Stratego Project provides a command line pretty printer ( http://releases.strategoxt.org/strategoxt-manual/unstable/manual/chunk-book/ref-pp-aterm.html )
for printing generic ATerm expressions.

In addition for dealing with highly annotated expressions the
hierarchal printer can be more human readable. (``+``) indicates
branching in the expression tree while (``key : value``) pairs
indicate annotations with metadata and type information.

.. code-block:: text

    + Appl
        type: (5, 5, int32, 5, 5, int32) -> 5, 5, int32 
        identity: 0
        elementwise: True
        associative: True
        idempotent: True
        inplace: False
        Term: abs
        Arguments:
          + Array
              type: 5, 5, int32
              local: True
              corder: True
              layout: chunked
          + Array
              type: 5, 5, int32
              local: True
              corder: True
              layout: chunked

Motivation
~~~~~~~~~~

ATerm is itself language agnostic, it is merely a human readable
AST format. For example the following C program can have many
different ATerm representations.

.. code-block:: c

    int main ()
    {
      int x;
      float y, z;

      x = 1;
      y = 2.0 / x;
      z = 3.0 + x;
      print(x, y, z);
    }

.. code-block:: python

    Program (
      [ Function(
          Type("int")
        , "main"
        , []
        , Body (
            [ Decl(Type("int"), ["x"])
            , Decl(Type("float"), ["y", "z"])
            ]
          , [ Assign("x", Const("1"))
            , Assign("y", Expr(Const("2.0"), BinOp("/"), Expr("x")))
            , Assign("z", Expr(Const("3.0"), BinOp("+"), Expr("x")))
            , Expr("print", [Expr("x"), Expr("y"), Expr("z")])
            ]
          )
        )
      ]
    )

Pattern matching using classical Visitor and expression traversal can be
quite verbose. Pattern matching allows us to abstract this logic down
into a more declarative form allowing us to build more powerful and
declarative tools for manipulating syntax trees.

::

    aterm = namedtuple('aterm', ('term', 'annotation'))
    astr  = namedtuple('astr', ('val',))
    aint  = namedtuple('aint', ('val',))
    areal = namedtuple('areal', ('val',))
    aappl = namedtuple('aappl', ('spine', 'args'))
    atupl = namedtuple('atupl', ('args'))
    aplaceholder = namedtuple('aplaceholder', ('type','args'))

    # Lets try and match f(x,y) using pure Python
    def match_simple(term):
        if isinstance(term, appl):
            if isinstance(term.args[0], aterm):
                if isinstance(term.args[1], aterm):
                    if term.args[0].term == 'x':
                        if term.args[1].term == 'y':
                            return True
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False

    # Compared to
    match('f(x,y)', 'f(x,y)')
