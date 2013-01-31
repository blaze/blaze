from pyparsing import Word, Regex, Optional, ZeroOrMore, \
        StringStart, StringEnd, alphas, alphanums, Or

# Parser to match the Blaze URL syntax
intNumber = Regex(r'[-+]?\b\d+\b')
arrayName = Regex(r'(/\w*)*[a-zA-Z0-9_]+\b')
bracketsIndexer = Optional(intNumber) + \
            Optional(':' + Optional(intNumber)) + \
            Optional(':' + Optional(intNumber))
indexerPattern = ('.' + Word(alphas + '_', alphanums + '_')) ^ \
        ('[' + bracketsIndexer + ZeroOrMore(',' + bracketsIndexer) + ']')
arrayBase = StringStart() + \
    arrayName + ZeroOrMore(indexerPattern) + \
    StringEnd()

def split_array_base(array_base):
    pieces = arrayBase.parseString(array_base)
    array_name = pieces[0]
    indexers = []
    i = 1
    while i < len(pieces):
        # Convert [...] into an int, a slice, or a tuple of int/slice
        if pieces[i] == '[':
            i += 1
            ilist = []
            while pieces[i-1] != ']':
                if pieces[i] != ':':
                    first = int(pieces[i])
                    i += 1
                else:
                    first = None
                if pieces[i] in [',', ']']:
                    i += 1
                    ilist.append(first)
                else:
                    i += 1
                    if pieces[i] not in [',', ':', ']']:
                        second = int(pieces[i])
                        i += 1
                    else:
                        second = None
                    if pieces[i] in [',', ']']:
                        i += 1
                        ilist.append(slice(first, second))
                    else:
                        i += 1
                        if pieces[i] not in [',', ']']:
                            third = int(pieces[i])
                            i += 1
                        else:
                            third = 1
                        ilist.append(slice(first, second, third))
                        i += 2
            if len(ilist) == 1:
                indexers.append(ilist[0])
            else:
                indexers.append(tuple(ilist))
        elif pieces[i] == '.':
            i += 1
        else:
            indexers.append(pieces[i])
            i += 1
            
    return array_name, indexers
    
