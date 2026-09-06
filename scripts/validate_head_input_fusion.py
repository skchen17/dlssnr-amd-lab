"""Supervised boundary/range gate for head gather + residual fusion."""
import argparse,sys,time,json,atexit,os,gc
from pathlib import Path
from validate_rocm_lifecycle import supervise,save

def child(a):
    import torch
    from native_head_torch import RecoveredHead32,compose_legacy_sdr_debug
    from native_head_fusion import HeadInput
    if not torch.version.hip or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('gfx1201 required')
    torch.set_num_threads(2);torch.manual_seed(137)
    op=HeadInput(a.dll,epilogue=a.epilogue,qkv=a.qkv);head=RecoveredHead32(bytes(21808)).eval().cuda()
    def phase(name):
        with (a.output/'phases.jsonl').open('a') as f:f.write(json.dumps({'phase':name,'pid':os.getpid()})+'\n')
    atexit.register(phase,'python_atexit')
    def wait():
        event=torch.cuda.Event();event.record();deadline=time.monotonic()+10
        while not event.query():
            if time.monotonic()>deadline:raise TimeoutError('GPU timeout; stop further work')
            time.sleep(.001)
    rows=[]
    with torch.no_grad():
        if a.qkv:
            def quant(x):return x.clamp(-448,448).to(torch.float8_e4m3fn).half()
            def norm(t):
                squares=(t*t).half()
                while squares.shape[-1]>1:squares=(squares[...,0::2]+squares[...,1::2]).half()
                inv=squares.float().clamp_min(6.198883056640625e-5).rsqrt().half()
                return (t*inv).half()
            all_bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16).cuda()
            for mode in ('all_bits','random'):
                projection=all_bits[:,None].expand(-1,96).contiguous().reshape(-1,64,96) if mode=='all_bits' else torch.randn(37,64,96,device='cuda',dtype=torch.float16)
                scale=torch.tensor(.713,device='cuda',dtype=torch.float16)
                q,k,v=projection.chunk(3,-1)
                expected_parts=[quant((norm(q)*scale).half()),quant(norm(k)),quant(v)]
                actual_parts=op.qkv(projection,scale);wait()
                for expected,actual in zip(expected_parts,actual_parts):
                    mask=~torch.isnan(expected)
                    if not torch.equal(torch.isnan(expected),torch.isnan(actual)) or not torch.equal(expected[mask].view(torch.int16),actual[mask].view(torch.int16)):
                        save(a.output/'qkv_mismatch.json',{'mode':mode,'non_nan_mismatches':int((expected[mask].view(torch.int16)!=actual[mask].view(torch.int16)).sum())})
                        raise RuntimeError('QKV normalization mismatch')
                rows.append({'qkv_mode':mode,'exact_non_nan':True})
            del all_bits,projection,scale,q,k,v,expected_parts,actual_parts,mask
        if a.epilogue:
            seed=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16).reshape(-1,32).cuda()
            product=torch.randn(seed.shape,device='cuda',dtype=torch.float32)
            scale=torch.randn(32,device='cuda',dtype=torch.float16)
            for factor in (None,scale):
                expected=((seed if factor is None else (seed*factor).half()).float()+product).half()
                actual=op.add(seed,product,factor);wait()
                mask=~torch.isnan(expected)
                if not torch.equal(torch.isnan(expected),torch.isnan(actual)) or not torch.equal(expected[mask].view(torch.int16),actual[mask].view(torch.int16)):
                    raise RuntimeError('head epilogue exhaustive seed mismatch')
            seed=torch.tensor([0.0,-0.0,0.0,-0.0],dtype=torch.float16,device='cuda')
            product=torch.tensor([0.0,0.0,-0.0,-0.0],dtype=torch.float32,device='cuda')
            expected=(seed.float()+product).half();actual=op.add(seed,product);wait()
            if not torch.equal(expected.view(torch.int16),actual.view(torch.int16)):raise RuntimeError('epilogue zero sign mismatch')
            rows.append({'epilogue_seed_patterns':65536,'scaled_and_unscaled':True,'signed_zero':True})
            del seed,product,scale,mask,factor
        head.main_scale.copy_(torch.randn_like(head.main_scale));head.skip_scale.copy_(torch.randn_like(head.skip_scale))
        for shape,indices in [((7,2048),head.block.a_index),((7,64,32),head.block.permutation)]:
            source=torch.randn(shape,device='cuda',dtype=torch.float16)
            expected=source[...,indices].clamp(-448,448).to(torch.float8_e4m3fn).half()
            actual=op.gather_e4(source,indices);wait()
            if not torch.equal(expected.view(torch.int16),actual.view(torch.int16)):raise RuntimeError('head gather quantize mismatch')
            rows.append({'gather_shape':shape,'bitwise_exact':True})
        del source,indices
        if a.c32_layout:
            from native_packed_swin import gather_packed,scatter_packed
            from native_c32_layout_fusion import c32_layout_fusion
            for w,h in [(4,4),(12,20),(128,64)]:
                for ox,oy in [(0,0),(-4,0),(0,-4),(-4,-4)]:
                    packed=torch.randn(w*h*32,device='cuda',dtype=torch.float16)
                    expected,mapping=gather_packed(packed,w,h,32,ox,oy)
                    with c32_layout_fusion(a.dll):actual,fmapping=gather_packed(packed,w,h,32,ox,oy)
                    wait()
                    if not torch.equal(expected.view(torch.int16),actual.view(torch.int16)):raise RuntimeError('C32 gather mismatch')
                    values=torch.randn(expected.shape[0],64,32,device='cuda',dtype=torch.float16)
                    expected=scatter_packed(values,mapping,w,h)
                    with c32_layout_fusion(a.dll):actual=scatter_packed(values,fmapping,w,h)
                    wait()
                    if not torch.equal(expected.view(torch.int16),actual.view(torch.int16)):raise RuntimeError('C32 scatter mismatch')
                    rows.append({'c32':True,'geometry':[w,h,ox,oy],'bitwise_exact':True})
            del packed,values,mapping,fmapping
        if a.pre_features:
            from native_preblock import single_color_features
            from native_pre_fusion import pre_features_fusion
            for w,h,pw,ph in [(1,1,8,8),(13,9,24,16),(127,63,128,128)]:
                color=torch.randn(h,w,4,device='cuda',dtype=torch.float16)
                options=dict(color_scale=1.7,conditioning=[0.0,1.0,-.5,.123,2.0])
                expected=single_color_features(color,pw,ph,19,**options)
                with pre_features_fusion(a.dll):actual=single_color_features(color,pw,ph,19,**options)
                wait()
                if not torch.equal(expected.view(torch.int16),actual.view(torch.int16)):raise RuntimeError('pre feature mismatch')
                rows.append({'pre':True,'geometry':[w,h,pw,ph],'bitwise_exact':True})
            del color
        for w,h in [(8,8),(24,16),(128,128),(640,384)]:
            main=torch.randn(w*h*8,device='cuda',dtype=torch.float16)
            skip=torch.randn(w*h*32,device='cuda',dtype=torch.float16)
            total=((w+11)//8)*((h+11)//8)
            for start,count in [(0,min(total,7)),(total-1,1),(max(0,total//2-2),min(5,total-total//2+2))]:
                ctas=torch.arange(start,start+count,device='cuda')
                expected=head.fuse(main,skip,ctas,w,h);actual=op(main,skip,head.main_scale,head.skip_scale,start,count,w,h);wait()
                exact=torch.equal(expected.view(torch.int16),actual.view(torch.int16))
                rows.append({'width':w,'height':h,'start':start,'count':count,'bitwise_exact':exact,
                    'max_abs':float((expected.float()-actual.float()).abs().max())})
                if not exact:
                    ix=(expected.view(torch.int16)!=actual.view(torch.int16)).nonzero()[:16]
                    save(a.output/'mismatch.json',{'cases':rows,'indices':ix.tolist(),
                         'expected':[float(expected[tuple(i)]) for i in ix],
                         'actual':[float(actual[tuple(i)]) for i in ix]})
                    raise RuntimeError('head input mismatch')
            for ow,oh in [(w,h),(w-3,h-5)]:
                base=torch.randn(oh,ow,4,device='cuda',dtype=torch.float16)
                residual=torch.randn(total,64,4,device='cuda',dtype=torch.float16)
                expected=compose_legacy_sdr_debug(residual,base,w,h);actual=op.compose(residual,base,w,h);wait()
                exact=torch.equal(expected.view(torch.int16),actual.view(torch.int16))
                rows.append({'compose':True,'width':ow,'height':oh,'bitwise_exact':exact})
                if not exact:raise RuntimeError('head compose mismatch')
    phase('gpu_work_complete')
    del main,skip,head,ctas,expected,actual,base,residual
    gc.collect();torch.cuda.empty_cache();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
    phase('resources_released')
    save(a.output/'child.json',{'checks_pass':True,'cases':rows,'authored_launches':op.launches,
        'allocated_after_release_bytes':torch.cuda.memory_allocated()})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--child',action='store_true');p.add_argument('--pre-features',action='store_true');p.add_argument('--c32-layout',action='store_true');p.add_argument('--epilogue',action='store_true');p.add_argument('--qkv',action='store_true');a=p.parse_args()
    if a.child:child(a)
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,str(Path(__file__).resolve()),'--child','--dll',str(a.dll.resolve()),'--output',str(a.output.resolve())]
        if a.pre_features:command.append('--pre-features')
        if a.c32_layout:command.append('--c32-layout')
        if a.epilogue:command.append('--epilogue')
        if a.qkv:command.append('--qkv')
        raise SystemExit(0 if supervise(command,a.output,timeout=90) else 2)
