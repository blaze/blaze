# -*- coding: utf-8 -*-

"""
Some Blaze AIR transformations and simplifications.
"""

from __future__ import print_function, division, absolute_import

from pykit.ir import Function, Builder, Value, Op

#------------------------------------------------------------------------
# Coercions -> Conversions
#------------------------------------------------------------------------

def explicit_coercions(func, env=None):
    """
    Turn implicit coercions into explicit conversion operations.
    """
    conversions = {}
    b = Builder(func)

    for op in func.ops:
        if op.opcode != 'kernel':
            continue

        overload = op.metadata['overload']
        signature = overload.resolved_sig
        parameters = signature.parameters[:-1]
        assert len(op.args) - 1 == len(parameters)

        # -------------------------------------------------
        # Identify conversion points

        replacements = {} # { arg : replacement_conversion }
        for arg, param_type in zip(op.args[1:], parameters):
            if arg.type != param_type:
                conversion = conversions.get((arg, param_type))
                if not conversion:
                    conversion = Op('convert', param_type, [arg])
                    b.position_after(arg)
                    b.emit(conversion)
                    conversions[arg, param_type] = conversion

                replacements[arg] = conversion

        # -------------------------------------------------

        op.replace_args(replacements)
