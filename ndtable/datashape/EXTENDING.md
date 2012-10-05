Pluggable Datashape
===================

The datashape system is designed to be extensible but in a way such
that all higher level datashape are neccessarily expressible in terms
of the lower datashapes. This is essential if we are to build a "data
description language that can be interpreted across machines and
possibly across different versions of Blaze.

This lets the domain expert describe their data in form that is
syntatically meaningful to their domain while not loosing the ability
for the compiler to understand it.

For example a Quaternion object can be represented as a 4x4 matrix of
integers in addition to some associated algebric laws.

```python
Quaternion = 4, 4, int32
```
