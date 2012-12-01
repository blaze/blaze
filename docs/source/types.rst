=====
Types
=====

.. Later this will be useful

.. graphviz::

   digraph foo {
      Dynamic;
      Primitive -> Null;
      Primitive -> Integer;

      DataShape -> Atom;
      DataShape -> CType;

      Atom -> Fixed;
      Atom -> TypeVar;
      Atom -> Var;
      Atom -> BitField;

      Atom -> Enum;
      Atom -> Union;

      DataShape -> Record;

      Atom -> Ptr;
   }

.. automodule:: ndtable.datashape.coretypes
   :members:
   :undoc-members:
