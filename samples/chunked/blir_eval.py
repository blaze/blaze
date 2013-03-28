
import blaze
from expression_builder import Visitor
from copy import deepcopy
import blaze.blir as blir
from time import time

class BlirEvaluator(object):
    """Evaluates expressions using blir

    Note that this is a sample, and has several assumptions
    hardcoded. This is 'proof-of-concept'
    """

    class _ExtractTerminals(Visitor):
        def accept_operation(self, node):
            return self.accept(node.lhs) |  self.accept(node.rhs)
           
        def accept_terminal(self, node):
            if isinstance(node.source, blaze.Array):
                return { node.source }
            else:
                return set()

    class _GenerateExpression(Visitor):
        def __init__(self, terms_dict):
            self.terms = terms_dict

        def accept_operation(self, node):
            a = self.accept(node.lhs)
            b = self.accept(node.rhs)
            return '(' + a + node.op + b + ')'
                
        def accept_terminal(self, node):
            if (isinstance(node.source, blaze.Array)):
                return self.terms[node.source] + '[i]'
            else:
                return repr(node.source)

    class _ParameterBinder(Visitor):
        def __init__(self, bind_dict):
            self.bind_dict = bind_dict

        def accept_operation(self, node):
            self.accept(node.lhs)
            self.accept(node.rhs)

        def accept_terminal(self, node):
            try:
                node.source = self.bind_dict[node.source]
            except KeyError:
                pass

    def __init__(self, root_node, operands=None):
        assert(root_node.op == 'dot')

        if operands:
            root_node = deepcopy(root_node)
            self._ParameterBinder(operands).accept(root_node)

        terms = self._ExtractTerminals().accept(root_node)
        terms = { obj: 'in%d' % i for i, obj in
                  enumerate(terms) }

        str_signature = self._gen_blir_signature(terms)
        str_lhs = self._GenerateExpression(terms).accept(root_node.lhs)
        str_rhs = self._GenerateExpression(terms).accept(root_node.rhs)
        code = '''
def main(%s, n: int) -> float {
    var float accum = 0.0;
    var int i = 0;
    for i in range(n) {
        accum = accum + (%s*%s);
    }
    return accum;
}
''' %  (str_signature, str_lhs, str_rhs)

        _, self.env = blir.compile(code)
        self.ctx = blir.Context(self.env)
        self.operands = list(terms)
        self.code = code
        self.time = 0.0

    def __del__(self):
        try:
            self.ctx.destroy()
        except:
            pass

    def eval(self, chunk_size=32768):
        operands = self.operands
        ctx = self.ctx
        execute = blir.execute
        
        total_size = operands[0].datashape.shape[-1].val
        offset = 0
        accum = 0.0
        while offset < total_size:
            curr_chunk_size = min(total_size - offset, chunk_size)
            slice_src = slice(offset, offset+curr_chunk_size)
            args = [op[slice_src] for op in operands]
            args.append(curr_chunk_size)
            t = time()
            accum += execute(ctx, args=args, fname='main')
            self.time += time() - t
            offset = slice_src.stop

        return accum

    
    def __str__(self):
        str_args = ',\n\n'.join([str(op) for op in  self.operands]) 
        return '''blir evaluator with code:\n%s\n\nwith args:\n%s\n''' % (self.code, str_args)

    @staticmethod
    def _to_blir_type_string(obj):
        if (isinstance(obj, blaze.Array)):
            p = obj.datashape.parameters
            assert(len(p) == 2)
            return 'array[%s]' % 'float'
        else:
            return 'float'

    @staticmethod
    def _gen_blir_decl(name, obj):
        return name + ': ' + BlirEvaluator._to_blir_type_string(obj)

    @staticmethod
    def _gen_blir_signature(terms):
        return ',\n\t'.join([BlirEvaluator._gen_blir_decl(pair[1], pair[0])
                             for pair in terms.iteritems()])

