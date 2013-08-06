import ctypes
from .. import llvm_array as lla
import llvm.core as lc
from llvm.core import Type, Function, Module
from ..llvm_array import (void_type, intp_type,
                SCALAR, POINTER, array_kinds, check_array,
                get_cpp_template, array_type, const_intp, LLArray, orderchar)
from ..py2help import PY2

int32_type = Type.int(32)
int8_p_type = Type.pointer(Type.int(8))
single_ckernel_func_type = Type.function(void_type,
                [int8_p_type, Type.pointer(int8_p_type), int8_p_type])

def map_llvm_to_ctypes(llvm_type, py_module=None, sname=None, i8p_str=False):
    '''
    Map an LLVM type to an equivalent ctypes type. py_module is an
    optional module that is used for structure wrapping.  If
    structures are found, the struct definitions will be created in
    that module.
    '''
    kind = llvm_type.kind
    if kind == lc.TYPE_INTEGER:
        ctype = getattr(ctypes,"c_int"+str(llvm_type.width))

    elif kind == lc.TYPE_DOUBLE:
        ctype = ctypes.c_double

    elif kind == lc.TYPE_FLOAT:
        ctype = ctypes.c_float

    elif kind == lc.TYPE_VOID:
        ctype = None

    elif kind == lc.TYPE_POINTER:
        pointee = llvm_type.pointee
        p_kind = pointee.kind
        if p_kind == lc.TYPE_INTEGER:
            width = pointee.width

            # Special case:  char * is mapped to strings
            if width == 8 and i8p_str:
                ctype = ctypes.c_char_p
            else:
                ctype = ctypes.POINTER(map_llvm_to_ctypes(pointee, py_module, sname))

        # Special case: void * mapped to c_void_p type
        elif p_kind == lc.TYPE_VOID:
            ctype = ctypes.c_void_p
        else:
            ctype = ctypes.POINTER(map_llvm_to_ctypes(pointee, py_module, sname))

    elif kind == lc.TYPE_ARRAY:
        ctype = llvm_type.count * map_llvm_to_ctypes(llvm_type.element, py_module, sname)
    elif kind == lc.TYPE_STRUCT:
        lookup = True
        if llvm_type.is_literal:
            if sname:
                struct_name = sname
            else:
                struct_name = 'llvm_struct'
                lookup = False
        else:
            struct_name = llvm_type.name
            struct_name = struct_name.replace('.','_')
        if PY2:
            struct_name = struct_name.encode('ascii')

        # If the named type is already known, return it
        if py_module and lookup:
            struct_type = getattr(py_module, struct_name, None)
        else:
            struct_type = None

        if struct_type and issubclass(struct_type, ctypes.Structure):
            return struct_type

        # If there is an object with the name of the structure already
        # present and it has the field names specified, use those names
        # to help out
        if hasattr(struct_type, '_fields_'):
            names = struct_type._fields_
        else:
            names = [ "e"+str(n) for n in range(llvm_type.element_count) ]

        # Create a class definition for the type. It is critical that this
        # Take place before the handling of members to avoid issues with
        # self-referential data structures
        if py_module and lookup:
            type_dict = { '__module__' : py_module.__name__}
        else:
            type_dict = {}
        ctype = type(ctypes.Structure)(struct_name, (ctypes.Structure,),
                                       type_dict)
        if py_module and lookup:
            setattr(py_module, struct_name, ctype)

        # Resolve the structure fields
        fields = [ (name, map_llvm_to_ctypes(elem, py_module))
                   for name, elem in zip(names, llvm_type.elements) ]

        # Set the fields member of the type last.  The order is critical
        # to deal with self-referential structures.
        setattr(ctype, '_fields_', fields)
    else:
        raise TypeError("Unknown type %s" % kind)
    return ctype

