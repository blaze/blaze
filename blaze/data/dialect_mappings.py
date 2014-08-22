from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


dialect_terms = {'delimiter': "delimiter", # Delimiter character (ALL)
                 'DELIMITER': "delimiter",
                 'sep': "delimiter",
                 'quotechar': "quotechar", # Single byte quote:  Default is double-quote. ex: "1997","Ford","E350")
                 'quote': "quotechar", 
                 'QUOTE': "quotechar",
                 'na_values': "na_value", # Null character -- Default: '' (empty string)
                 'null': "na_value",
                 'NULL': "na_value",
                 'escapechar': "escapechar", # A single one-byte character for defining escape characters.
                 'ESCAPECHAR': "escapechar", # Default is the same as the QUOTE
                 'ESCAPE': "escapechar", 
                 'escape': "escapechar",
                 'lineterminator': "lineterminator", #  string used to terminate lines. default is r'\n'
                 'LINETERMINATOR': "lineterminator",
                 'skiprows': "skiprows", # Integer value for number of rows to skip
                 'SKIPROWS': "skiprows",
                 'ignorerows': "skiprows", 
                 'IGNOREROWS': "skiprows",
                 'format': "format_str", # format is a restricted word in python
                 'FORMAT': "format_str",
                 'header': "header", # Boolean Flag (POSTGRES,MySQL) to define if file contains a header
                 'HEADER': "header",
                 'encoding': "encoding", # Charatcter Encoding -- POSTGRES: utf (default),MySQL latin1 (default)
                 'ENCODING': "encoding",
                 }
