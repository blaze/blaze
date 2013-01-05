# dlex.py. This file automatically created by PLY (version 3.4). Don't edit!
_tabversion   = '3.4'
_lextokens    = {'LBRACE': 1, 'NAME': 1, 'SEMI': 1, 'SPACE': 1, 'NUMBER': 1, 'EQUALS': 1, 'COMMA': 1, 'COLON': 1, 'TYPE': 1, 'RBRACE': 1}
_lexreflags   = 0
_lexliterals  = '=,():{}'
_lexstateinfo = {'INITIAL': 'inclusive'}
_lexstatere   = {'INITIAL': [('(?P<t_TYPE>type)|(?P<t_SPACE>\\s)|(?P<t_NUMBER>\\d+)|(?P<t_NAME>[a-zA-Z_][a-zA-Z0-9_]*)|(?P<t_RBRACE>\\})|(?P<t_LBRACE>\\{)|(?P<t_COLON>:)|(?P<t_COMMA>,)|(?P<t_SEMI>;)|(?P<t_EQUALS>=)', [None, ('t_TYPE', 'TYPE'), ('t_SPACE', 'SPACE'), ('t_NUMBER', 'NUMBER'), (None, 'NAME'), (None, 'RBRACE'), (None, 'LBRACE'), (None, 'COLON'), (None, 'COMMA'), (None, 'SEMI'), (None, 'EQUALS')])]}
_lexstateignore = {'INITIAL': '\n'}
_lexstateerrorf = {'INITIAL': 't_error'}
