"""One bounded actual-feature decoder subgraph GPU run under host supervision."""
import json
from pathlib import Path
import time
import torch
from validate_native_decoder_pyramid import load_fixture
from run_swin1h_native_validation import compare,digest,checked
from native_swin_torch import encode_e4
from native_execution_policy import current_profile
from validate_rocm_lifecycle import ORIGINAL_SHA


def exercise_decoder(phase,wait,properties):
    if current_profile()!='native_fp16':
        raise ValueError('this reviewed decoder probe requires native_fp16')
    model,entry,skips,references,records,input_hashes=load_fixture()
    model.cuda()
    entry=entry.cuda()
    skips={c:value.cuda() for c,value in skips.items()}
    if any(t.device.type!='cuda' for t in [entry,*skips.values(),*model.parameters(),*model.buffers()]):
        raise RuntimeError('CPU neural tensor in decoder GPU subgraph')
    wait()
    phase('decoder_inputs_and_weights_ready')
    stages=[]
    with torch.no_grad():
        iterator=iter(model.stages(entry,skips,320,192))
        for index in range(22):
            started=time.perf_counter()
            block,value=next(iterator)
            finite=torch.isfinite(value).all()
            wait()
            host_ms=(time.perf_counter()-started)*1000
            if not bool(finite):
                raise RuntimeError('nonfinite decoder GPU output; stop without retry')
            actual=encode_e4(value)
            wait()
            raw=actual.cpu().numpy().tobytes()
            comparison=compare(references[block],raw)
            stages.append({'block':block,'slot':block+84,'host_submit_and_wait_ms':host_ms,
                           'comparison_diagnostic':comparison,'output_sha256':digest(raw),
                           'original_record_sha256':digest(records[block])})
            phase(f'decoder_block{block}_complete')
        # Finish generator normally so no suspended frame holds its tensors.
        try:
            next(iterator)
            raise RuntimeError('unexpected extra decoder stage')
        except StopIteration:
            pass
    phase('gpu_work_complete')
    cpu_root=Path('results/20260906_native_decoder_pyramid_cpu_v1')
    cpu_spec=json.loads((cpu_root/'manifest.json').read_bytes())
    if (cpu_spec['arithmetic_profile']!='native_fp16' or cpu_spec['original_model_sha256']!=ORIGINAL_SHA
            or cpu_spec['external_input_hashes']!=input_hashes):
        raise ValueError('CPU decoder comparison provenance differs')
    cpu_raw=checked(cpu_root/'slot153.e4',cpu_spec['stages'][-1]['output_sha256'])
    return {'name':'decoder_pyramid','checks_pass':all(s['comparison_diagnostic']['nonfinite']==0 for s in stages),
            'backend':'pytorch_rocm','device':properties.name,'architecture':properties.gcnArchName,
            'torch':torch.__version__,'hip':torch.version.hip,'original_model_sha256':ORIGINAL_SHA,
            'external_input_hashes':input_hashes,'encoder_skip_source':'ORIGINAL_CAPTURED_EXTERNAL_INPUTS',
            'decoder_interior_rtx_substitution':False,'stages':stages,'output_sha256':digest(raw),
            'cpu_candidate_comparison':compare(cpu_raw,raw),
            'host_stage_sum_ms':sum(s['host_submit_and_wait_ms'] for s in stages),
            'timing_scope':'single decoder feature subgraph; per-stage waits; not game or complete-image FPS',
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'cpu_neural_fallback':False,'backward_tested':False,'training_started':False,
            'native_graph_complete':False,'image_quality_verified':False}
