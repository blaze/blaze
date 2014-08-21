from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import inspect

# Delimiter character (ALL)
delimiter_str = "delimiter"
dialect_terms = {'delimiter': delimiter_str, 'DELIMITER': delimiter_str, 'sep': delimiter_str,}

# Single byte quote:  Default is double-quote. ex: "1997","Ford","E350")
quotechar_str = "quotechar"
dialect_terms.update({'quotechar': quotechar_str, 'quote': quotechar_str, 'QUOTE': quotechar_str})

# Null character -- Default: '' (empty string)
na_str = "na_values"
dialect_terms.update({'na_values': na_str, 'null': na_str, 'NULL': na_str,})

# A single one-byte character for defining escape characters. Default is the same as the QUOTE
esc_str = "escapechar"
dialect_terms.update({'escapechar': esc_str, 'ESCAPECHAR': esc_str, 'ESCAPE': esc_str, 'escape': esc_str,})

#  string used to terminate lines. default is r'\n'
lineterminator_str = "lineterminator"
dialect_terms.update({'lineterminator': lineterminator_str, 'LINETERMINATOR': lineterminator_str})


# Interger value for number of rows to skip
skiprows_str = "skiprows"
dialect_terms.update({'skiprows': skiprows_str, 'SKIPROWS': skiprows_str,
                      'ignorerows': skiprows_str, 'IGNOREROWS': skiprows_str,
                      })

# format is a restricted word in python
format_str = "format_str"
dialect_terms.update({'format': format_str, 'FORMAT': format_str,})

# Boolean Flag (POSTGRES) to define if file contains a header
header_str = "header"
dialect_terms.update({'header': header_str, 'HEADER': header_str,})
