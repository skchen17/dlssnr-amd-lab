"""Experimental shape-specific ROCm graph replay, not D3D12/game interop.

No neural CPU fallback, image readback, skipped frames, old-output reuse or
resolution changes. A new geometry requires a newly prepared graph. Caller
owns GPU completion waiting; an exception never implies GPU cancellation.
"""
import threading
import torch
from native_execution_policy import execution_policy


def validate_input(actual, prepared):
    if (actual.shape != prepared.shape or actual.dtype != prepared.dtype
            or actual.device != prepared.device):
        raise ValueError('graph input shape/dtype/device changed; prepare new resources')


class CapturedTensorRunner:
    game_runtime_ready = False

    def __init__(self, forward, example, *, precision, wait):
        if not torch.version.hip or example.device.type != 'cuda' or example.dtype != torch.float16:
            raise ValueError('native ROCm FP16 GPU input required')
        self.owner = threading.get_ident()
        self.stream_id = torch.cuda.current_stream().cuda_stream
        self.closed = False
        self.sequence = 0
        self.input = example.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        # Establish GEMM handles/workspaces on the exact capture stream first.
        with torch.no_grad(), execution_policy(precision), torch.cuda.stream(side):
            warm = forward(self.input)
            wait()
            del warm
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), execution_policy(precision), torch.cuda.graph(self.graph, stream=side):
            self.output = forward(self.input)

    def require_owner(self):
        if self.closed or threading.get_ident() != self.owner or torch.cuda.current_stream().cuda_stream != self.stream_id:
            raise RuntimeError('closed graph or changed owner/stream')

    def submit(self, source):
        self.require_owner()
        validate_input(source, self.input)
        with torch.no_grad():
            self.input.copy_(source)
            self.graph.replay()
        self.sequence += 1
        # Output is GPU storage overwritten by next submit. Consume only after
        # completion and before submitting another frame; no CPU copy is made.
        return self.output

    def close(self, wait):
        self.require_owner()
        wait()
        self.graph.reset()
        del self.output, self.input, self.graph
        self.closed = True
