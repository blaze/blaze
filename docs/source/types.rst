==========
Core Types
==========

Relations of Types
~~~~~~~~~~~~~~~~~~

Arrows denotes commensurability, not hierarchy.

.. TODO: These are wicked old... fix

.. graphviz::

   digraph relation {
        Type -> Fixed;
        Type -> CType;
        Type -> Top;
        Type -> Dynamic;

        Type -> Sum;
        Type -> Product;
        Type -> Enum;

        Sum -> Either;
        Sum -> Union;
        Sum -> Range;
        Sum -> Option;

        Range -> Stream;
   }

Structural Relations
~~~~~~~~~~~~~~~~~~~~

.. graphviz::

   digraph structural {
        Type -> Unit;
        Type -> Aggregate;

        Unit -> Fixed;
        Unit -> CType;
        Unit -> Top;
        Unit -> Dynamic;
        Unit -> Null;

        Aggregate -> Sum;
        Aggregate -> Product;
        Aggregate -> Record;
        Aggregate -> Enum;
   }

.. graphviz::

   digraph structural_unit {
        Top -> Dimension;
        Top -> Measure;

        Dimension -> Fixed;
        Measure -> CType;
   }

Implementation
~~~~~~~~~~~~~~

.. graphviz::

   digraph top {
      Dynamic;
      Primitive -> Null;
      Primitive -> Integer;

      DataShape -> Atom;
      DataShape -> CType;

      Atom -> Fixed;
      Atom -> TypeVar;
      Atom -> Range;

      Atom -> Enum;
      Atom -> Union;

      DataShape -> Record;
   }


API
~~~

.. automodule:: blaze.datashape.coretypes
   :members:
   :undoc-members:
