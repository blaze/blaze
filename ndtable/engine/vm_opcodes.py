# Binary
# $OP $1 $2

# Unary
# $OP $1

# Function
# $OP $OP $1...

ops = [
      'FUNC_MAP'
    , 'FUNC_REDUCE'
    , 'FUNC_ACCUMULATE'

    , 'UNARY_POSITIVE'
    , 'UNARY_NEGATIVE'
    , 'UNARY_NOT'
    , 'UNARY_CONVERT'
    , 'UNARY_INVERT'
    , 'BINARY_POWER'
    , 'BINARY_MULTIPLY'
    , 'BINARY_DIVIDE'
    , 'BINARY_MODULO'
    , 'BINARY_ADD'
    , 'BINARY_SUBTRACT'
    , 'BINARY_ALL'
    , 'BINARY_ANY'

    , 'INDEX_ELEMENT'
    , 'INDEX_SLICE'
    , 'INDEX_FANCY'

    , 'BOUNDSCHECK'
    , 'PROMOTE'
]

# Types not in this map would require a promotion operation at
# the VM level

binary_types = [
    '??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
    'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'gg->g', 'FF->F',
    'DD->D', 'GG->G', 'Mm->M', 'mm->m', 'mM->M', 'OO->O'
]

unary_types = [
    '?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
    'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G',
    'M->M', 'm->m', 'O->O'
]
