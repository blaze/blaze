from collections import namedtuple

RuleNode     = namedtuple( 'Rule', ('label', 'lhs', 'rhs'))
StrategyNode = namedtuple('Strategy', ('label', 'combinator', 'args'))
AsNode       = namedtuple('As', ('tag', 'pattern'))
