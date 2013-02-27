from pprint import pformat

def dump_instance(sig, outsig):
    print "Signature Instance".center(80, '=')
    print 'signature', sig.show()
    print 'instance', outsig
    print "".center(80, '=')

def debug_aliases(derived, newcls, sym, alias):
    print 'Class', derived.name, newcls.params
    print 'Symbol', sym
    print 'Alias', alias
    print "".center(80, '=')

def debug_adhoc(anon_refs):
    D = 4
    print "Ad-hoc".center(80, '=')
    print pformat(anon_refs.items(), depth=D, width=1)
    print "".center(80, '=')

def debug_bound(bound_ns):
    D = 4
    print "Bound".center(80, '=')
    print pformat([(a, [(c, d.show()) for c,d in b.items()])
        for a,b in bound_ns.iteritems()],
        depth=D, width=1
    )
    print "".center(80, '=')

def debug_classes(root):
    for clsname, cls in root:
        print cls.name
        for defn in cls.defs:
            print '\t', 'fun:', defn

        for alias in cls.aliases:
            print '\t', 'alias:', alias, '~', cls.aliases[alias]
