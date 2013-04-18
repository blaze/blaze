# _dlex.py. This file automatically created by PLY (version 3.4). Don't edit!
_tabversion   = '3.4'
_lextokens    = {'NAME': 1, 'INT': 1, 'DOUBLE': 1, 'INCOMB': 1, 'AS': 1, 'ARROW': 1, 'STRING': 1}
_lexreflags   = 0
_lexliterals  = ';,|:=(){}[]'
_lexstateinfo = {'INITIAL': 'inclusive'}
_lexstatere   = {'INITIAL': [('(?P<t_newline>\\n+)|(?P<t_DOUBLE>\\d+\\.(\\d+)?)|(?P<t_INT>\\d+)|(?P<t_ARROW>->)|(?P<t_AS>\\@)|(?P<t_STRING>"([^"\\\\]|\\\\.)*")|(?P<t_COMMENT>\\#.*)|(?P<t_NAME>[a-zA-Z_][a-zA-Z0-9_]*)|(?P<t_INCOMB>fail|id|\\<\\+|\\;)', [None, ('t_newline', 'newline'), ('t_DOUBLE', 'DOUBLE'), None, ('t_INT', 'INT'), ('t_ARROW', 'ARROW'), ('t_AS', 'AS'), ('t_STRING', 'STRING'), None, ('t_COMMENT', 'COMMENT'), (None, 'NAME'), (None, 'INCOMB')])]}
_lexstateignore = {'INITIAL': ' \t\r'}
_lexstateerrorf = {'INITIAL': 't_error'}
