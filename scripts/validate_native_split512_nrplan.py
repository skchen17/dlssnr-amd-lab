"""One-window fixed-sequence resident C512 split-block NRPlan gate."""
from __future__ import annotations

import argparse, atexit, ctypes as ct, gc, json, os, sys
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled
from native_cpp_nr_plan import (ApproxSplit512Block, ArenaRegion, BindingsV3, Desc,
                                ModelPackageStats, ResourceStats, ShapeDescV3,
                                approximate_split512_descriptor_array)
from validate_native_stage_nrplan import bind, call, metric, sha256
from validate_rocm_lifecycle import save, supervise


def select(path):
    topology=json.loads(path.read_text(encoding='utf-8'))
    approximate_split512_descriptor_array(topology)
    rows=[row for row in topology['blocks'] if row['record_number']==23]
    if len(rows)!=1:raise ValueError('missing unique C512 record 23')
    return rows[0]


def descriptor(row):
    header={'struct_size','windows','feature_width','feature_height','origin_x',
            'origin_y','record_number','reserved'}
    fields=[name for name,_ in ApproxSplit512Block._fields_ if name not in header]
    return ApproxSplit512Block(ct.sizeof(ApproxSplit512Block),1,4,4,0,0,
        row['record_number'],0,*(row[name] for name in fields))


def exercise(args,row):
    import torch
    from native_model_package import load_package
    from native_packed_swin import gather_packed,logical_indices,scatter_packed
    from native_split_swin512 import RecoveredSplitSwin512,RECORD_SIZES
    from native_swin_torch import encode_e4,quantize_e4
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm GPU required')
    props=torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(1_500_000_000/props.total_memory)
    records,_,_=load_package('local_models/native_single_color_v1',
        'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    parts={name:records[(23,index)] for index,name in enumerate(RECORD_SIZES)}
    model=RecoveredSplitSwin512(**parts).eval().cuda()
    torch.manual_seed(23512)
    logical=quantize_e4((torch.randn(4*4*512,device='cuda',dtype=torch.float16)*.1)).contiguous()
    initial=encode_e4(logical).contiguous()
    windows,mapping=gather_packed(logical,4,4,512,0,0)
    ai,ri=logical_indices(512);ai=ai.cuda();ri=ri.cuda()
    with torch.no_grad():
        output=model(windows[:,ai],windows[:,ri])
        expected=encode_e4(quantize_e4(scatter_packed(output,mapping,4,4))).contiguous()
    arena=json.loads(args.arena.read_text(encoding='utf-8'))
    regions=[ArenaRegion(r['offset'],r['bytes'],r['alignment'],0,r['name'].encode())
             for r in arena['regions']]
    region_array=(ArenaRegion*len(regions))(*regions)
    native=descriptor(row)
    library=ct.CDLL(str(args.dll.resolve()));bind(library)
    library.nrPlanConfigureApproxSplit512Blocks.argtypes=[ct.c_void_p,
        ct.POINTER(ApproxSplit512Block),ct.c_uint32]
    library.nrPlanConfigureApproxSplit512Blocks.restype=ct.c_int
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
        shape=ShapeDescV3(ct.sizeof(ShapeDescV3),4,4,4,4,0,0,4,4,0,1,2,3)
        call(library.nrPlanPrepareShape(handle,ct.byref(shape)),'shape')
        call(library.nrPlanConfigureApproxSplit512Blocks(handle,ct.byref(native),1),
             'configure split')
        call(library.nrPlanInitializeArenaFromDevice(handle,row['raw_resident_offset'],
             ct.c_void_p(initial.data_ptr()),initial.numel()),'input')
        call(library.nrPlanFinalize(handle),'finalize')
        binding=BindingsV3(ct.sizeof(BindingsV3),3,dummy.data_ptr(),dummy.data_ptr(),
            0,0,0,0,0,0,4,4,4,4,0,0,4,4,0.,0.,1.,1.,1.,1.,0,1,1,0)
        call(library.nrPlanSubmitV3(handle,ct.byref(binding),None,None),'submit')
        entries,post,qkv,scratch=(ct.c_uint64() for _ in range(4))
        call(library.nrPlanDebugGetFixedBoundaryState(handle,ct.byref(entries),
             ct.byref(post),ct.byref(qkv),ct.byref(scratch)),'state')
        scalars=64*512
        call(library.nrPlanDebugCopyFixedBoundaryScratchToDevice(handle,scalars*7,
             ct.c_void_p(actual.data_ptr()),actual.numel()),'output')
        error=metric(actual,expected);resource=ResourceStats()
        call(library.nrPlanGetResourceStats(handle,ct.byref(resource)),'resources')
        approximate=bool(error['finite'] and error['nrmse']<=.12 and
                         error['max_absolute_error']<=.25)
        return {'checks_pass':approximate and entries.value==post.value==qkv.value==1,
            'approximate_split512_gate_pass':approximate,'record':23,
            'geometry':[4,4,512],'error_vs_reference':error,
            'state':{'entries':entries.value,'post':post.value,'qkv':qkv.value,
                     'scratch_bytes':scratch.value},
            'resource':{'workspace_bytes':resource.workspace_bytes,
                        'weight_bytes':resource.weight_bytes,
                        'deployment_ready':bool(resource.deployment_ready)},
            'device':props.name,'architecture':props.gcnArchName,
            'gpu_work_cancelled':False,'automatic_retry':False}
    finally:
        if handle.value:call(library.nrPlanDestroy(handle),'destroy')


def child(args):
    require_gpu_tests_enabled('native C512 resident split gate');row=select(args.topology)
    def phase(name):
        with (args.output/'phases.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'phase':name,'pid':os.getpid()})+'\n')
    atexit.register(phase,'python_atexit');phase('child_enter');import torch
    try:result=exercise(args,row);phase('gpu_work_complete')
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
    for name in ('dll','package','arena','topology','output'):
        p.add_argument(f'--{name}',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--child',action='store_true')
    a=p.parse_args();row=select(a.topology)
    preflight={'status':'CPU_PREFLIGHT_PASS_GPU_NOT_EXECUTED','dll_sha256':sha256(a.dll),
               'record':row['record_number'],'geometry':[4,4,512]}
    if not a.execute:save(a.output,preflight);return True
    require_gpu_tests_enabled('native C512 resident split gate')
    if a.child:return child(a)
    a.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--child','--execute']
    for name in ('dll','package','arena','topology','output'):
        command += [f'--{name}',str(getattr(a,name).resolve())]
    return supervise(command,a.output,timeout=120)


if __name__=='__main__':raise SystemExit(0 if main() else 2)
