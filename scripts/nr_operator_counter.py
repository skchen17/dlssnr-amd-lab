"""ATen invocation counts, explicitly NOT HIP dispatch or GPU kernel counts."""
from collections import Counter,defaultdict
from torch.utils._python_dispatch import TorchDispatchMode


class OperatorCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.block=-1;self.counts=Counter();self.blocks=defaultdict(Counter)

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        name=str(func);self.counts[name]+=1;self.blocks[self.block][name]+=1
        return func(*args,**(kwargs or {}))

    def report(self):
        return {'scope':'ATen invocations, NOT GPU dispatch count; includes views and host metadata operations',
                'total':sum(self.counts.values()),'operators':dict(self.counts.most_common()),
                'blocks':{str(k):dict(v.most_common()) for k,v in self.blocks.items()},
                'gpu_dispatch_count':None}
