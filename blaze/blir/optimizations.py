import llvm.ee as le
import llvm.passes as lp

from llvm.workaround.avx_support import detect_avx_support

#------------------------------------------------------------------------
# Optimizer
#------------------------------------------------------------------------

class LLVMOptimizer(object):
    inline_threshold = 1000

    def __init__(self, module, opt_level=3, loop_vectorize=True):
        # opt_level is used for both module level (opt) and
        # instruction level optimization (cg) for TargetMachine
        # and PassManager

        if not detect_avx_support():
            tm = le.TargetMachine.new(
                opt = opt_level,
                cm  = le.CM_JITDEFAULT,
                features='-avx',
            )
        else:
            tm = le.TargetMachine.new(
                opt = opt_level,
                cm  = le.CM_JITDEFAULT,
                features='' ,
            )

        pass_opts = dict(
            fpm = False,
            mod = module,
            opt = opt_level,
            vectorize = False,
            loop_vectorize = loop_vectorize,
            inline_threshold=self.inline_threshold,
        )

        pms = lp.build_pass_managers(tm = tm, **pass_opts)
        pms.pm.run(module)
