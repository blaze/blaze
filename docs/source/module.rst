=============
Module System
=============

A goal of the Blaze module system is to package together packets of
definitions of data types and associated operations over that type and
enforce a consistent naming scheme for these definitions.

The module system also serves to define the signatures for determining
types over the defined modules to define operations in a generic way.

Typesets
--------

+-----------+----------------------------------------------------+
| array     |  Array T                                           |
|-----------+----------------------------------------------------|
| table     |  Table T                                           |
|-----------+----------------------------------------------------|
| ints      |  int8, int16, int32, int64                         |
|-----------+----------------------------------------------------|
| uints     |  uint8, uint16, uint32, uint64                     |
|-----------+----------------------------------------------------|
| floats    |  float32, float64                                  |
|-----------+----------------------------------------------------|
| complexes |  complex64, complex128                             |
|-----------+----------------------------------------------------|
| bools     |  bool                                              |
|-----------+----------------------------------------------------|
| string    |  String, Varchar, Blob                             |
|-----------+----------------------------------------------------|
| discrete  |  ints , uints                                      |
|-----------+----------------------------------------------------|
| reals     |  ints , floats                                     |
|-----------+----------------------------------------------------|
| continuous|  floats , complexes                                |
|-----------+----------------------------------------------------|
| numeric   |  discrete , continuous                             |
|-----------+----------------------------------------------------|
| indexable |  array_like, table_like                            |
|-----------+----------------------------------------------------|
| universal |  top , numeric , indexable , string                |
+-----------+----------------------------------------------------+
