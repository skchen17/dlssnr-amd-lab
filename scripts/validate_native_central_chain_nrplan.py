"""One-submit C512 encoder -> ViT8 -> C512 decoder resident-chain gate."""
from __future__ import annotations

import argparse, atexit, ctypes as ct, gc, json, os, sys
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled
from native_cpp_nr_plan import (ApproxBottleneck, ApproxSplit512Block, ApproxVitBlock,
                                ArenaRegion, BindingsV3, Desc, ModelPackageStats,
                                ResourceStats, ShapeDescV3,
                                approximate_bottleneck_descriptor,
                                approximate_split512_descriptor_array,
                                approximate_vit_descriptor_array)
from validate_native_stage_nrplan import bind, call, metric, sha256
from validate_rocm_lifecycle import save, supervise


def load_descriptors(args):
    split=approximate_split512_descriptor_array(json.loads(
        args.split_topology.read_text(encoding='utf-8')))
    vit=approximate_vit_descriptor_array(json.loads(
        args.vit_topology.read_text(encoding='utf-8')))
    bottleneck=approximate_bottleneck_descriptor(json.loads(
        args.bottleneck_topology.read_text(encoding='utf-8')))
    return split,vit,bottleneck


def reference(records,split_rows):
    import torch
    from native_multiscale import pack_image,unpack_image
    from native_packed_swin import gather_packed,logical_indices,scatter_packed
    from native_split_image512 import pool_to_bottleneck
    from native_split_swin512 import RecoveredSplitSwin512,RECORD_SIZES
    from native_swin_torch import encode_e4,quantize_e4
    from native_transition_projections import EncoderFinalProjection,DecoderInputProjection
    from native_vit1024 import RecoveredVit1024,RECORD_SIZES as VIT_PARTS

    torch.manual_seed(512310)
    logical=quantize_e4((torch.randn(36,60,512,device='cuda',dtype=torch.float16)*.1))
    value=pack_image(logical).contiguous();initial=encode_e4(value).contiguous()
    ai,ri=logical_indices(512);ai=ai.cuda();ri=ri.cuda()
    with torch.no_grad():
        for number in range(23,31):
            row=split_rows[number]
            parts={name:records[(number,index)] for index,name in enumerate(RECORD_SIZES)}
            block=RecoveredSplitSwin512(**parts).eval().cuda()
            windows,mapping=gather_packed(value,60,36,512,row['origin_x'],row['origin_y'])
            value=quantize_e4(scatter_packed(block(windows[:,ai],windows[:,ri]),
                                             mapping,60,36)).contiguous()
        skip=value
        pooled=pool_to_bottleneck(unpack_image(value,60,36,512))
        value=EncoderFinalProjection(records[(30,4)]).eval().cuda()(pooled)
        value=value.reshape(1,640,1024)
        for number in range(31,39):
            parts={name:records[(number,index)] for index,name in enumerate(VIT_PARTS)}
            value=RecoveredVit1024(**parts).eval().cuda()(value,query_chunk=32)
        value=DecoderInputProjection(records[(39,0)]).eval().cuda().fuse(
            value[0].reshape(20,32,1024),unpack_image(skip,60,36,512),60,36)
        value=pack_image(value).contiguous()
        for number in range(40,48):
            row=split_rows[number]
            parts={name:records[(number,index)] for index,name in enumerate(RECORD_SIZES)}
            block=RecoveredSplitSwin512(**parts).eval().cuda()
            windows,mapping=gather_packed(value,60,36,512,row['origin_x'],row['origin_y'])
            value=quantize_e4(scatter_packed(block(windows[:,ai],windows[:,ri]),
                                             mapping,60,36)).contiguous()
    return initial,encode_e4(value).contiguous()


def exercise(args,descriptors):
    import torch
    from native_model_package import load_package
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm GPU required')
    props=torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(2_500_000_000/props.total_memory)
    records,_,_=load_package('local_models/native_single_color_v1',
        'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    split_rows={row['record_number']:row for row in json.loads(
        args.split_topology.read_text(encoding='utf-8'))['blocks']}
    initial,expected=reference(records,split_rows)
    split,vit,bottleneck=descriptors
    arena=json.loads(args.arena.read_text(encoding='utf-8'))
    regions=[ArenaRegion(r['offset'],r['bytes'],r['alignment'],0,r['name'].encode())
             for r in arena['regions']]
    region_array=(ArenaRegion*len(regions))(*regions)
    library=ct.CDLL(str(args.dll.resolve()));bind(library)
    library.nrPlanConfigureApproxSplit512Blocks.argtypes=[ct.c_void_p,
        ct.POINTER(ApproxSplit512Block),ct.c_uint32]
    library.nrPlanConfigureApproxVitBlocks.argtypes=[ct.c_void_p,
        ct.POINTER(ApproxVitBlock),ct.c_uint32]
    library.nrPlanConfigureApproxBottleneck.argtypes=[ct.c_void_p,ct.POINTER(ApproxBottleneck)]
    library.nrPlanDebugGetFixedCentralState.argtypes=[ct.c_void_p]+[ct.POINTER(ct.c_uint64)]*5
    library.nrPlanDebugCopyFixedResidentToDevice.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64]
    for name in ('nrPlanConfigureApproxSplit512Blocks','nrPlanConfigureApproxVitBlocks',
                 'nrPlanConfigureApproxBottleneck','nrPlanDebugGetFixedCentralState',
                 'nrPlanDebugCopyFixedResidentToDevice'):
        getattr(library,name).restype=ct.c_int
    handle=ct.c_void_p();dummy=torch.empty(4,dtype=torch.float16,device='cuda')
    actual=torch.empty_like(expected)
    try:
        call(library.nrPlanCreate(ct.byref(Desc(arena['workspace_bytes'],0,1920,1080)),
                                  ct.byref(handle)),'create')
        call(library.nrPlanSetPrecisionProfile(handle,1),'profile')
        call(library.nrPlanSetExecutionMode(handle,1),'fixed sequence')
        stats=ModelPackageStats();call(library.nrPlanLoadModelPackage(handle,
            str(args.package.resolve()).encode(),ct.byref(stats)),'load package')
        call(library.nrPlanConfigureArena(handle,region_array,len(regions)),'arena')
        shape=ShapeDescV3(ct.sizeof(ShapeDescV3),1920,1080,1920,1080,0,0,1920,1080,
                          0,1,2,3)
        call(library.nrPlanPrepareShape(handle,ct.byref(shape)),'shape')
        call(library.nrPlanConfigureApproxSplit512Blocks(handle,split,len(split)),'split')
        call(library.nrPlanConfigureApproxVitBlocks(handle,vit,len(vit)),'vit')
        call(library.nrPlanConfigureApproxBottleneck(handle,ct.byref(bottleneck)),'bottleneck')
        call(library.nrPlanInitializeArenaFromDevice(handle,split[0].raw_resident_offset,
            ct.c_void_p(initial.data_ptr()),initial.numel()),'input')
        call(library.nrPlanFinalize(handle),'finalize')
        binding=BindingsV3(ct.sizeof(BindingsV3),3,dummy.data_ptr(),dummy.data_ptr(),
            0,0,0,0,0,0,1920,1080,1920,1080,0,0,1920,1080,
            0.,0.,1.,1.,1.,1.,0,1,1,0)
        call(library.nrPlanSubmitV3(handle,ct.byref(binding),None,None),'submit')
        call(library.nrPlanDebugCopyFixedResidentToDevice(handle,
             ct.c_void_p(actual.data_ptr()),actual.numel()),'copy resident')
        enc,blocks,dec,vit_bytes,skip_bytes=(ct.c_uint64() for _ in range(5))
        call(library.nrPlanDebugGetFixedCentralState(handle,ct.byref(enc),ct.byref(blocks),
             ct.byref(dec),ct.byref(vit_bytes),ct.byref(skip_bytes)),'central state')
        entries,post,qkv,scratch=(ct.c_uint64() for _ in range(4))
        call(library.nrPlanDebugGetFixedBoundaryState(handle,ct.byref(entries),
             ct.byref(post),ct.byref(qkv),ct.byref(scratch)),'boundary state')
        error=metric(actual,expected);resource=ResourceStats()
        call(library.nrPlanGetResourceStats(handle,ct.byref(resource)),'resources')
        approximate=bool(error['finite'] and error['nrmse']<=.25 and
                         error['max_absolute_error']<=.5)
        return {'checks_pass':approximate and enc.value==dec.value==1 and blocks.value==8
                and entries.value==post.value==qkv.value==16,
            'approximate_central_chain_gate_pass':approximate,
            'records':[23,47],'geometry':[60,36,512],'vit_tokens':640,
            'error_vs_reference':error,
            'central_state':{'encoder_chains':enc.value,'vit_block_chains':blocks.value,
                'decoder_chains':dec.value,'vit_scratch_bytes':vit_bytes.value,
                'bottleneck_skip_bytes':skip_bytes.value},
            'stage_state':{'entries':entries.value,'post':post.value,'qkv':qkv.value,
                           'scratch_bytes':scratch.value},
            'resource':{'workspace_bytes':resource.workspace_bytes,
                        'weight_bytes':resource.weight_bytes,
                        'deployment_ready':bool(resource.deployment_ready)},
            'device':props.name,'architecture':props.gcnArchName,
            'gpu_work_cancelled':False,'automatic_retry':False}
    finally:
        if handle.value:call(library.nrPlanDestroy(handle),'destroy')


def child(args):
    require_gpu_tests_enabled('native central resident chain gate');descriptors=load_descriptors(args)
    def phase(name):
        with (args.output/'phases.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'phase':name,'pid':os.getpid()})+'\n')
    atexit.register(phase,'python_atexit');phase('child_enter');import torch
    try:result=exercise(args,descriptors);phase('gpu_work_complete')
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
    for name in ('dll','package','arena','split-topology','vit-topology',
                 'bottleneck-topology','output'):
        p.add_argument(f'--{name}',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--child',action='store_true')
    a=p.parse_args();load_descriptors(a)
    preflight={'status':'CPU_PREFLIGHT_PASS_GPU_NOT_EXECUTED','dll_sha256':sha256(a.dll),
               'records':[23,47],'split_blocks':16,'vit_blocks':8,'vit_tokens':640}
    if not a.execute:save(a.output,preflight);return True
    require_gpu_tests_enabled('native central resident chain gate')
    if a.child:return child(a)
    a.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--child','--execute']
    for name in ('dll','package','arena','split_topology','vit_topology',
                 'bottleneck_topology','output'):
        command += ['--'+name.replace('_','-'),str(getattr(a,name).resolve())]
    return supervise(command,a.output,timeout=240)


if __name__=='__main__':raise SystemExit(0 if main() else 2)
