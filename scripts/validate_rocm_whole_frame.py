"""Bounded single-color GPU chain; not a temporal/HDR/game acceptance test."""
import json
from pathlib import Path
import time
import torch
from native_capture_fixture import whole_frame_fixture
from native_whole_frame import SingleColorWholeFrame
from native_execution_policy import current_profile
from run_swin1h_native_validation import digest,checked
from describe_native_reconstruction import ORIGINAL_SHA


def exercise_whole_frame(phase,wait,properties,name,iterations=1,artifact_output=None):
    if current_profile()!='native_fp16':
        raise ValueError('reviewed whole-frame probe requires native_fp16')
    cpu_root=Path('results/20260906_single_color_chain128_v1' if name=='whole_frame128'
                  else 'results/20260906_single_color_gamecrop640_v1')
    report=json.loads((cpu_root/'manifest.json').read_bytes())
    if report['original_model_sha256']!=ORIGINAL_SHA or report['arithmetic_profile']!='native_fp16':
        raise ValueError('CPU comparison model/profile differs')
    w,h=report['input_geometry']
    raw=checked(cpu_root/'input.rgba16f',report['input_sha256'])
    records,origins,settings=whole_frame_fixture()
    model=SingleColorWholeFrame(records,origins,**settings).cuda()
    color=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(h,w,4).cuda()
    if any(t.device.type!='cuda' for t in [color,*model.parameters(),*model.buffers()]):
        raise RuntimeError('CPU neural tensor in GPU chain')
    wait(); phase('whole_frame_inputs_and_weights_ready')
    stages=[]; frames=[]
    with torch.no_grad():
        for frame in range(iterations):
            frame_stages=[]
            iterator=iter(model.stages(color,0))
            for index in range(71):
                start=time.perf_counter()
                b,value=next(iterator)
                finite=torch.isfinite(value).all()
                wait()
                ms=(time.perf_counter()-start)*1000
                if b!=index or not bool(finite):
                    raise RuntimeError('missing or nonfinite GPU stage; stop without retry')
                frame_stages.append({'block':b,'shape':list(value.shape),'host_submit_and_wait_ms':ms})
                phase(f'whole_frame_iteration{frame}_block{b}_complete')
            try:
                next(iterator)
                raise RuntimeError('unexpected extra stage')
            except StopIteration:
                pass
            actual=value.cpu().float()
            frames.append({'iteration':frame,'output_sha256':digest(actual.half().numpy().tobytes()),
                           'host_stage_sum_ms':sum(s['host_submit_and_wait_ms'] for s in frame_stages),
                           'allocated_bytes':torch.cuda.memory_allocated()})
            if artifact_output is not None:
                with (artifact_output/'completed_frames.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(frames[-1],allow_nan=False)+'\n')
            if frame==0:
                stages=frame_stages
    phase('gpu_work_complete')
    reference=torch.frombuffer(bytearray(checked(cpu_root/'output.rgba16f',report['output_sha256'])),
                               dtype=torch.float16).reshape(h,w,4).float()
    delta=actual[...,:3]-reference[...,:3]
    changed=actual[...,:3]-color.cpu().float()[...,:3]
    # Readback is diagnostic AFTER neural execution, never a game frame pathway.
    output=actual.half().numpy().tobytes()
    if artifact_output is not None:
        with (artifact_output/'output.rgba16f').open('xb') as stream:
            stream.write(output)
    repeat_exact=len({f['output_sha256'] for f in frames})==1
    return {'name':name,'checks_pass':repeat_exact,'backend':'pytorch_rocm','cpu_neural_fallback':False,
            'device':properties.name,'architecture':properties.gcnArchName,'hip':torch.version.hip,
            'original_model_sha256':ORIGINAL_SHA,'input_sha256':digest(raw),'output_sha256':digest(output),
            'record_hashes':{f'block{b}.layer{l}.layer':digest(r) for (b,l),r in records.items()},
            'input_source':report['input_source'],'input_geometry':[w,h],
            'all_encoder_skips_generated':True,'rtx_intermediate_inputs':False,
            'stages':stages,'host_stage_sum_ms':sum(s['host_submit_and_wait_ms'] for s in stages),
            'iterations':iterations,'frames':frames,'identical_input_repeat_exact':repeat_exact,
            'timing_scope':'single cold staged debug frame, not optimized game FPS',
            'cpu_candidate_rgb_rmse':float(delta.square().mean().sqrt()),
            'cpu_candidate_rgb_max_error':float(delta.abs().max()),
            'input_output_rgb_mae':float(changed.abs().mean()),
            'output_clipped_zero_fraction':float((actual[...,:3]==0).float().mean()),
            'output_clipped_one_fraction':float((actual[...,:3]==1).float().mean()),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'single_color_tensor_chain_connected':True,'native_graph_complete':False,
            'composition':'LEGACY_SDR_DIAGNOSTIC_ONLY','temporal_inputs_supported':False,'hdr_supported':False,
            'game_runtime_ready':False,'image_quality_verified':False,'training_started':False}
