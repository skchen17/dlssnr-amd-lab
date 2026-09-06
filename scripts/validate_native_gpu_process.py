"""Standalone two-process native texture loop. Not an injection or game acceptance."""
import argparse
import atexit
import ctypes as ct
import gc
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from native_gpu_protocol import parse,MAX_MESSAGE_BYTES,worker_identity
from native_model_package import sha
from validate_rocm_lifecycle import save,supervise,configure_fault_logging


def observed_parent(pid):
    """Read Windows process table; do not trust a self-reported parent PID."""
    class Entry(ct.Structure):
        _fields_=[('size',ct.c_uint32),('usage',ct.c_uint32),('pid',ct.c_uint32),('heap',ct.c_size_t),
                  ('module',ct.c_uint32),('threads',ct.c_uint32),('parent',ct.c_uint32),
                  ('priority',ct.c_long),('flags',ct.c_uint32),('exe',ct.c_wchar*260)]
    k=ct.WinDLL('kernel32',use_last_error=True)
    k.CreateToolhelp32Snapshot.argtypes=[ct.c_uint32,ct.c_uint32];k.CreateToolhelp32Snapshot.restype=ct.c_void_p
    k.Process32FirstW.argtypes=[ct.c_void_p,ct.POINTER(Entry)];k.Process32FirstW.restype=ct.c_int
    k.Process32NextW.argtypes=[ct.c_void_p,ct.POINTER(Entry)];k.Process32NextW.restype=ct.c_int
    k.CloseHandle.argtypes=[ct.c_void_p];k.CloseHandle.restype=ct.c_int
    snapshot=k.CreateToolhelp32Snapshot(2,0)
    if snapshot==ct.c_void_p(-1).value:raise ct.WinError(ct.get_last_error())
    try:
        entry=Entry();entry.size=ct.sizeof(entry);ok=k.Process32FirstW(snapshot,ct.byref(entry))
        while ok:
            if entry.pid==pid:return entry.parent
            ok=k.Process32NextW(snapshot,ct.byref(entry))
        raise ValueError('reported worker process not found')
    finally:k.CloseHandle(snapshot)


class WorkerPipe:
    def __init__(self,command,root):
        self.stderr=(root/'worker_stderr.log').open('x')
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
                                      text=True,encoding='utf-8',bufsize=1,env=dict(os.environ,PYTHONNOUSERSITE='1'))
        save(root/'worker_process.json',{'pid':self.process.pid,'command':command,'automatic_kill':False})
        self.messages=queue.Queue()
        self.worker_pid=None;self.root=root
        def reader():
            try:
                while True:
                    line=self.process.stdout.readline(MAX_MESSAGE_BYTES+1)
                    if not line:raise RuntimeError('worker closed stdout')
                    self.messages.put(parse(line))
            except BaseException as e:self.messages.put(e)
        threading.Thread(target=reader,daemon=True).start()
    def send(self,value):
        line=json.dumps(value,allow_nan=False)
        if len(line)>MAX_MESSAGE_BYTES:raise ValueError('metadata too large')
        self.process.stdin.write(line+'\n');self.process.stdin.flush()
    def receive(self,op):
        try:value=self.messages.get(timeout=30)
        except queue.Empty:raise TimeoutError('worker response timeout; no kill, retry or GPU cancellation')
        if isinstance(value,BaseException):raise value
        with (self.root/'metadata_received.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        if op=='hello' and self.worker_pid is None:
            actual=value.get('pid');parent=observed_parent(actual) if type(actual) is int and actual>0 else None
            self.worker_pid=worker_identity(self.process.pid,actual,parent,self.process.poll() is None)
            save(self.root/'worker_identity.json',{'launcher_pid':self.process.pid,'worker_pid':actual,'observed_parent':parent,'verified':True})
        if value.get('op')!=op or value.get('pid',self.worker_pid)!=self.worker_pid:raise ValueError('worker identity/response mismatch')
        return value
    def close(self):
        self.process.stdin.close()
        code=self.process.wait(timeout=15)
        self.stderr.close()
        if code:raise RuntimeError('worker did not exit normally')


def exercise(a,phase):
    import torch
    from native_bridge_harness import SharedTextureHarness
    torch.set_num_threads(2)
    if not torch.version.hip:raise RuntimeError('ROCm only')
    if a.network:
        root=Path('results/20260906_single_color_chain128_v1' if a.width==128 else 'results/20260906_single_color_gamecrop640_v1')
        spec=json.loads((root/'manifest.json').read_bytes());w,h=spec['input_geometry'];raw=(root/'input.rgba16f').read_bytes()
        if sha(raw)!=spec['input_sha256']:raise ValueError('baseline input hash')
    else:
        w,h=a.width,129;host=(torch.arange(h*w*4).reshape(h,w,4)%2048).half()/2048;raw=host.numpy().tobytes()
    flipped=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(h,w,4).flip(1).numpy().tobytes()
    if torch.cuda.is_initialized():raise RuntimeError('producer must not initialize HIP')
    command=[sys.executable,str(Path(__file__).with_name('native_gpu_worker.py').resolve()),'--producer',str(os.getpid()),
             '--dll',str(a.dll.resolve()),'--output',str(a.output.resolve()),'--max-frames',str(a.iterations),'--diagnostic-readback',
             '--optimization-profile',a.optimization_profile]
    if a.network:command.append('--network')
    worker=WorkerPipe(command,a.output);hello=worker.receive('hello')
    luid=hello.get('adapter_luid')
    if type(luid) is not int or not 0<luid<2**64:raise ValueError('invalid worker adapter identity')
    bridge=SharedTextureHarness(a.dll,w,h,producer_only=True,adapter_luid=luid);lib=bridge.library;phase('gpu_initialized')
    lib.nr_bridge_export.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_uint64,ct.POINTER(ct.c_uint64)];lib.nr_bridge_export.restype=ct.c_int
    lib.nr_bridge_remote_done.argtypes=[ct.c_void_p,ct.c_uint64];lib.nr_bridge_remote_done.restype=ct.c_int
    values=(ct.c_uint64*16)();bridge.check(lib.nr_bridge_export(bridge.handle,worker.worker_pid,1,values))
    worker.send({'op':'prepare','descriptor':list(values)});ready=worker.receive('ready')
    if ready.get('optimization_profile')!=a.optimization_profile:raise ValueError('worker schedule mismatch')
    phase('remote_import_and_model_ready')
    frames=[]
    for index in range(a.iterations):
        if index:bridge.next_frame()
        source=raw if index%2==0 else flipped
        bridge.upload_fixture(source);phase(f'frame{index}_producer_signaled')
        worker.send({'op':'frame','generation':1,'sequence':index+1,'seed':0});result=worker.receive('done')
        if result.get('generation')!=1 or result.get('sequence')!=index+1 or result.get('input_sha256')!=sha(source) or result.get('finite') is not True:
            raise ValueError('worker input/frame association differs')
        if a.network and a.optimization_profile=='native_opt3' and (
                result.get('allocator_budget_bytes')!=5000000000 or not 0<result.get('peak_reserved_bytes',0)<=5000000000):
            raise ValueError('worker allocator memory budget differs or exceeded')
        bridge.check(lib.nr_bridge_remote_done(bridge.handle,index+1))
        actual=bridge.read_fixture()
        if sha(actual)!=result.get('output_sha256'):raise RuntimeError('game-side fixture consumer differs from remote GPU result')
        if not a.network and actual!=(flipped if index%2==0 else raw):raise RuntimeError('transport identity differs')
        worker.send({'op':'consumed','generation':1,'sequence':index+1});released=worker.receive('released')
        if released.get('generation')!=1 or released.get('sequence')!=index+1:raise ValueError('stale release acknowledgement')
        frames.append(result);phase(f'frame{index}_consumer_completed_and_lease_released')
        with (a.output/'completed_frames.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    worker.send({'op':'close'});closed=worker.receive('closed');worker.close();phase('worker_normal_exit')
    bridge.close();phase('gpu_work_complete')
    with (a.output/'output.rgba16f').open('xb') as f:f.write(actual)
    exact_baseline=None
    if a.network:
        baseline=Path('results/20260906_rocm_whole640_monitorv2_hundred/output.rgba16f') if w==640 else root/'output.rgba16f'
        exact_baseline=frames[0]['output_sha256']==sha(baseline.read_bytes())
    stable=all(len({f['output_sha256'] for f in frames if f['sequence']%2==parity})==1 for parity in ({1} if a.iterations==1 else {0,1}))
    changes=a.iterations==1 or frames[0]['output_sha256']!=frames[1]['output_sha256']
    return {'checks_pass':stable and changes and (not a.network or w!=640 or exact_baseline is True) and closed.get('allocated_after_release_bytes')==0 and closed.get('reserved_after_release_bytes')==0,
            'optimization_profile':a.optimization_profile,
            'mode':'network' if a.network else 'transport','backend':'pytorch_rocm','input_geometry':[w,h],
            'two_processes':os.getpid()!=worker.worker_pid,'producer_pid':os.getpid(),'worker_pid':worker.worker_pid,
            'metadata_only_ipc':True,'native_gpu_to_gpu_pixels':True,'worker_normal_exit':True,
            'producer_hip_initialized':torch.cuda.is_initialized(),
            'worker_release':closed,'frames':frames,'same_candidate_first_frame_exact':exact_baseline,
            'alternating_outputs_stable':stable,'changed_input_changes_output':changes,
            'bridge_dll_sha256':sha(a.dll.read_bytes()),'game_runtime_ready':False,'pre_hud_hook_verified':False,
            'cpu_neural_fallback':False,'diagnostic_readback':True,'training_started':False}


def child(a):
    trace=(a.output/'traceback.log').open('x');configure_fault_logging(trace);globals()['_trace']=trace
    def phase(name):
        with (a.output/'phases.jsonl').open('a') as f:f.write(json.dumps({'phase':name,'pid':os.getpid(),'monotonic':time.monotonic()})+'\n')
    atexit.register(phase,'python_atexit');phase('child_enter')
    names={p.name for p in Path(__file__).parent.glob('native_*.py')}|{'validate_native_gpu_process.py'}
    sources={n:sha((Path(__file__).parent/n).read_bytes()) for n in sorted(names)}
    save(a.output/'source_snapshot.json',sources)
    result=exercise(a,phase);gc.collect()
    import torch
    result['allocated_after_release_bytes']=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0
    result['reserved_after_release_bytes']=torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0
    result['source_changes_during_run']=[n for n,s in sources.items() if sha((Path(__file__).parent/n).read_bytes())!=s]
    result['checks_pass'] &= not result['source_changes_during_run'] and result['reserved_after_release_bytes']==0
    phase('resources_released');save(a.output/'child.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--network',action='store_true');p.add_argument('--iterations',type=int,choices=(1,12),default=1)
    p.add_argument('--optimization-profile',choices=('baseline12','native_opt2','native_opt3'),default='baseline12')
    p.add_argument('--width',type=int,choices=(128,640,641),default=128);p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.network and a.width==641:p.error('641 network not reviewed')
    if a.child:child(a)
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,str(Path(__file__).resolve()),'--child','--dll',str(a.dll.resolve()),'--output',str(a.output.resolve()),'--width',str(a.width),'--iterations',str(a.iterations),'--optimization-profile',a.optimization_profile]
        if a.network:command.append('--network')
        raise SystemExit(0 if supervise(command,a.output,timeout=90) else 2)
