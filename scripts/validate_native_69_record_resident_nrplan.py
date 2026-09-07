"""One-submit 1080p record1..69 resident NRPlan functional gate (no Pre/Head)."""
from __future__ import annotations

import argparse, atexit, ctypes as ct, gc, json, os, sys
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled
from native_cpp_nr_plan import (ApproxBottleneck, ApproxScaleTransition,
    ApproxSplit512Block, ApproxStageBlock, ApproxVitBlock, ArenaRegion, BindingsV3,
    Desc, GraphStats, ModelPackageStats, PerformanceStats, ResourceStats, ShapeDescV3,
    approximate_bottleneck_descriptor, approximate_scale_transition_descriptor_array,
    approximate_split512_descriptor_array, approximate_stage_descriptor_array,
    approximate_vit_descriptor_array)
from validate_native_stage_nrplan import bind,call,sha256
from validate_rocm_lifecycle import save,supervise


def descriptors(args):
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    return (approximate_stage_descriptor_array(load(args.stage_topology)),
            approximate_split512_descriptor_array(load(args.split_topology)),
            approximate_vit_descriptor_array(load(args.vit_topology)),
            approximate_bottleneck_descriptor(load(args.bottleneck_topology)),
            approximate_scale_transition_descriptor_array(load(args.transition_topology)))


def exercise(args,values):
    import torch
    from native_multiscale import pack_image
    from native_swin_torch import encode_e4,quantize_e4
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm GPU required')
    props=torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(2_000_000_000/props.total_memory)
    torch.manual_seed(16969)
    initial=encode_e4(pack_image(quantize_e4(torch.randn(576,960,32,device='cuda',
        dtype=torch.float16)*.05))).contiguous()
    output=torch.empty_like(initial)
    stage,split,vit,bottleneck,transitions=values
    arena=json.loads(args.arena.read_text(encoding='utf-8'))
    regions=[ArenaRegion(r['offset'],r['bytes'],r['alignment'],0,r['name'].encode())
             for r in arena['regions']]
    region_array=(ArenaRegion*len(regions))(*regions)
    library=ct.CDLL(str(args.dll.resolve()));bind(library)
    signatures=(
        ('nrPlanConfigureApproxSplit512Blocks',[ct.c_void_p,ct.POINTER(ApproxSplit512Block),ct.c_uint32]),
        ('nrPlanConfigureApproxVitBlocks',[ct.c_void_p,ct.POINTER(ApproxVitBlock),ct.c_uint32]),
        ('nrPlanConfigureApproxBottleneck',[ct.c_void_p,ct.POINTER(ApproxBottleneck)]),
        ('nrPlanConfigureApproxScaleTransitions',[ct.c_void_p,ct.POINTER(ApproxScaleTransition),ct.c_uint32]),
        ('nrPlanDebugGetFixedTransitionState',[ct.c_void_p]+[ct.POINTER(ct.c_uint64)]*4),
        ('nrPlanDebugGetFixedCentralState',[ct.c_void_p]+[ct.POINTER(ct.c_uint64)]*5),
        ('nrPlanDebugCopyFixedResidentToDevice',[ct.c_void_p,ct.c_void_p,ct.c_uint64]))
    for name,types in signatures:
        getattr(library,name).argtypes=types;getattr(library,name).restype=ct.c_int
    handle=ct.c_void_p();dummy=torch.empty(4,dtype=torch.float16,device='cuda')
    try:
        call(library.nrPlanCreate(ct.byref(Desc(arena['workspace_bytes'],0,1920,1080)),
                                  ct.byref(handle)),'create')
        call(library.nrPlanSetPrecisionProfile(handle,1),'profile')
        call(library.nrPlanSetExecutionMode(handle,1),'fixed sequence')
        library.nrPlanDebugSetFixedHostBoundaries.argtypes=[ct.c_void_p,ct.c_uint8]
        library.nrPlanDebugSetFixedHostBoundaries.restype=ct.c_int
        if args.async_boundaries:
            call(library.nrPlanDebugSetFixedHostBoundaries(handle,0),'disable host boundaries')
        package=ModelPackageStats();call(library.nrPlanLoadModelPackage(handle,
            str(args.package.resolve()).encode(),ct.byref(package)),'package')
        call(library.nrPlanConfigureArena(handle,region_array,len(regions)),'arena')
        shape=ShapeDescV3(ct.sizeof(ShapeDescV3),1920,1080,1920,1080,0,0,1920,1080,0,1,2,3)
        call(library.nrPlanPrepareShape(handle,ct.byref(shape)),'shape')
        call(library.nrPlanConfigureApproxStageBlocks(handle,stage,len(stage),0),'stage')
        call(library.nrPlanConfigureApproxSplit512Blocks(handle,split,len(split)),'split')
        call(library.nrPlanConfigureApproxVitBlocks(handle,vit,len(vit)),'vit')
        call(library.nrPlanConfigureApproxBottleneck(handle,ct.byref(bottleneck)),'bottleneck')
        call(library.nrPlanConfigureApproxScaleTransitions(handle,transitions,len(transitions)),
             'transitions')
        call(library.nrPlanInitializeArenaFromDevice(handle,stage[0].raw_resident_offset,
             ct.c_void_p(initial.data_ptr()),initial.numel()),'input')
        call(library.nrPlanFinalize(handle),'finalize')
        graph=GraphStats();call(library.nrPlanGetGraphStats(handle,ct.byref(graph)),'graph stats')
        binding=BindingsV3(ct.sizeof(BindingsV3),3,dummy.data_ptr(),dummy.data_ptr(),
            0,0,0,0,0,0,1920,1080,1920,1080,0,0,1920,1080,0.,0.,1.,1.,1.,1.,0,1,1,0)
        call(library.nrPlanSubmitV3(handle,ct.byref(binding),None,None),'submit')
        call(library.nrPlanDebugCopyFixedResidentToDevice(handle,
             ct.c_void_p(output.data_ptr()),output.numel()),'output')
        entries,post,qkv,scratch=(ct.c_uint64() for _ in range(4))
        call(library.nrPlanDebugGetFixedBoundaryState(handle,ct.byref(entries),ct.byref(post),
             ct.byref(qkv),ct.byref(scratch)),'stage state')
        enc,dec,target,skips=(ct.c_uint64() for _ in range(4))
        call(library.nrPlanDebugGetFixedTransitionState(handle,ct.byref(enc),ct.byref(dec),
             ct.byref(target),ct.byref(skips)),'transition state')
        benc,vblocks,bdec,vscratch,bskip=(ct.c_uint64() for _ in range(5))
        call(library.nrPlanDebugGetFixedCentralState(handle,ct.byref(benc),ct.byref(vblocks),
             ct.byref(bdec),ct.byref(vscratch),ct.byref(bskip)),'central state')
        perf=PerformanceStats();call(library.nrPlanGetPerformanceStats(handle,ct.byref(perf)),'perf')
        resource=ResourceStats();call(library.nrPlanGetResourceStats(handle,ct.byref(resource)),'resources')
        decoded=output.view(torch.float8_e4m3fn).half();finite=bool(torch.isfinite(decoded).all())
        nonzero=int(torch.count_nonzero(output).item())
        output_hash=__import__('hashlib').sha256(output.cpu().numpy().tobytes()).hexdigest().upper()
        sync_hash='B97CDE1719F1FEAEC0167E14490F8DAD412116629EF49B05E5EB72C9FCD9C3A1'
        checks=finite and nonzero>0 and entries.value==post.value==qkv.value==60 and \
            enc.value==dec.value==4 and benc.value==bdec.value==1 and vblocks.value==8
        if args.async_boundaries:checks&=output_hash==sync_hash
        return {'checks_pass':checks,'functional_resident_69_record_gate_pass':checks,
            'scope':'records 1..69, excludes Pre record0 and Head record70',
            'output_finite':finite,'output_nonzero_bytes':nonzero,
            'output_sha256':output_hash,'sync_baseline_sha256':sync_hash,
            'hash_matches_sync_baseline':output_hash==sync_hash,
            'host_boundaries_enabled':not args.async_boundaries,
            'stage_state':{'entries':entries.value,'post':post.value,'qkv':qkv.value,
                           'scratch_bytes':scratch.value},
            'transition_state':{'encoder_chains':enc.value,'decoder_chains':dec.value,
                'target_bytes':target.value,'skip_pool_bytes':skips.value},
            'central_state':{'encoder_chains':benc.value,'vit_block_chains':vblocks.value,
                'decoder_chains':bdec.value,'vit_scratch_bytes':vscratch.value,
                'bottleneck_skip_bytes':bskip.value},
            'diagnostic_event_span_ms':perf.last_gpu_ms,
            'graph':{'total_nodes':graph.total_nodes,'kernel_nodes':graph.kernel_nodes,
                     'memcpy_nodes':graph.memcpy_nodes,
                     'owns_graph_source':bool(resource.owns_graph_source),
                     'references_external_allocations':bool(
                         resource.graph_references_external_allocations)},
            'resource':{'workspace_bytes':resource.workspace_bytes,'weight_bytes':resource.weight_bytes,
                        'deployment_ready':bool(resource.deployment_ready)},
            'device':props.name,'architecture':props.gcnArchName,
            'gpu_work_cancelled':False,'automatic_retry':False}
    finally:
        if handle.value:call(library.nrPlanDestroy(handle),'destroy')


def child(args):
    require_gpu_tests_enabled('69-record resident NRPlan functional gate');values=descriptors(args)
    def phase(name):
        with (args.output/'phases.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'phase':name,'pid':os.getpid()})+'\n')
    atexit.register(phase,'python_atexit');phase('child_enter');import torch
    try:result=exercise(args,values);phase('gpu_work_complete')
    except BaseException as error:
        result={'checks_pass':False,'error_type':type(error).__name__,'error':str(error),
                'gpu_work_cancelled':False,'automatic_retry':False};phase('gpu_work_failed')
    gc.collect();torch.cuda.empty_cache()
    if hasattr(torch._C,'_cuda_clearCublasWorkspaces'):torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache();result['allocated_after_release_bytes']=torch.cuda.memory_allocated()
    result['reserved_after_release_bytes']=torch.cuda.memory_reserved()
    result['checks_pass']&=result['allocated_after_release_bytes']==0 and \
        result['reserved_after_release_bytes']==0
    phase('resources_released');save(args.output/'child.json',result);return result['checks_pass']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dll','package','arena','stage-topology','split-topology','vit-topology',
                 'bottleneck-topology','transition-topology','output'):
        p.add_argument(f'--{name}',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--child',action='store_true')
    p.add_argument('--async-boundaries',action='store_true')
    a=p.parse_args();values=descriptors(a)
    preflight={'status':'CPU_PREFLIGHT_PASS_GPU_NOT_EXECUTED','dll_sha256':sha256(a.dll),
        'standard_blocks':len(values[0]),'split_blocks':len(values[1]),
        'vit_blocks':len(values[2]),'transitions':len(values[4]),'records':[1,69]}
    if not a.execute:save(a.output,preflight);return True
    require_gpu_tests_enabled('69-record resident NRPlan functional gate')
    if a.child:return child(a)
    a.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--child','--execute']
    for name in ('dll','package','arena','stage_topology','split_topology','vit_topology',
                 'bottleneck_topology','transition_topology','output'):
        command += ['--'+name.replace('_','-'),str(getattr(a,name).resolve())]
    if a.async_boundaries:command.append('--async-boundaries')
    return supervise(command,a.output,timeout=240)


if __name__=='__main__':raise SystemExit(0 if main() else 2)
