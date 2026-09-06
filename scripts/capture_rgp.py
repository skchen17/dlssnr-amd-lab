"""Run one bounded HIP target under Radeon GPU Profiler and preserve evidence.

The AMD tool package is intentionally external to the repository.  This
wrapper never changes TDR, persistent clock settings, driver experiments, or
the target command.  RDP's temporary profiling clocks are restored by RDP and
its log is retained for verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import queue
import threading
import time
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled


def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):digest.update(block)
    return digest.hexdigest().upper()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdp",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--process",default="python.exe");parser.add_argument("--dispatch-start",type=int,default=1)
    parser.add_argument("--dispatch-count",type=int,required=True);parser.add_argument("--delay-ms",type=int,default=8000)
    parser.add_argument("--ready-token",default="")
    parser.add_argument("--timeout-seconds",type=int,default=120);parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    require_gpu_tests_enabled('RGP capture',profiler=True)
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    if not command or min(args.dispatch_start,args.dispatch_count,args.timeout_seconds)<=0 or not 0<=args.delay_ms<=30000:
        parser.error("bounded capture settings and a target command are required")
    args.output.mkdir(parents=True,exist_ok=False)
    rdp=args.rdp.resolve();profile=(args.output/"capture.rgp").resolve()
    if not rdp.is_file():raise FileNotFoundError(rdp)
    rdp_command=[str(rdp),"--mode","profiling","--process",args.process]
    if args.ready_token:
        if args.dispatch_start!=1:parser.error("ready-token capture starts at the next dispatch; dispatch-start must be 1")
        rdp_command += ["--rgp-capture-mode","dispatch","--rgp-render-op-count",str(args.dispatch_count)]
    else:
        rdp_command += ["--rgp-auto-capture",f"dispatch:{args.dispatch_start}:{args.dispatch_count}",
                        "--rgp-auto-capture-delay",str(args.delay_ms)]
    rdp_command += ["--rgp-counter-collection","--output",str(profile),"--verbose"]
    started=time.monotonic();target=None;rdp_process=None
    with ((args.output/"rdp.stdout.log").open("x") as rdp_out,
          (args.output/"rdp.stderr.log").open("x") as rdp_err,
          (args.output/"target.stdout.log").open("x") as target_out,
          (args.output/"target.stderr.log").open("x") as target_err):
        rdp_process=subprocess.Popen(rdp_command,stdin=subprocess.PIPE,stdout=rdp_out,stderr=rdp_err,text=True,
                                     env=dict(os.environ,PYTHONNOUSERSITE="1"))
        time.sleep(1)
        if args.ready_token:
            target=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=target_err,text=True,bufsize=1,
                                    env=dict(os.environ,PYTHONNOUSERSITE="1",PYTHONUNBUFFERED="1"))
            lines=queue.Queue()
            def copy_stdout():
                assert target is not None and target.stdout is not None
                for line in target.stdout:
                    target_out.write(line);target_out.flush();lines.put(line)
            reader=threading.Thread(target=copy_stdout,daemon=True);reader.start()
            deadline=time.monotonic()+min(45,args.timeout_seconds);ready=False
            while time.monotonic()<deadline and target.poll() is None:
                try:line=lines.get(timeout=.1)
                except queue.Empty:continue
                if args.ready_token in line:ready=True;break
            if ready and rdp_process.poll() is None and rdp_process.stdin:
                rdp_process.stdin.write("c\n");rdp_process.stdin.flush()
            if not ready:target_out.write("CAPTURE_WRAPPER_READY_TOKEN_NOT_SEEN\n");target_out.flush()
        else:
            ready=None
            target=subprocess.Popen(command,stdout=target_out,stderr=target_err,
                                    env=dict(os.environ,PYTHONNOUSERSITE="1",PYTHONUNBUFFERED="1"))
        try:target_code=target.wait(timeout=args.timeout_seconds);target_timeout=False
        except subprocess.TimeoutExpired:target_code=None;target_timeout=True
        if target_timeout:
            # A host timeout does not prove that queued GPU work was cancelled.
            # Preserve both processes for manual review and do not retry.
            rdp_code=None;rdp_timeout=True
        else:
            # Interactive captures remain in the panel command loop after a
            # successful dump. Once the finite target has exited, ask that
            # profiler process to quit cleanly; never terminate queued GPU work.
            if (args.ready_token or target_code) and rdp_process.poll() is None and rdp_process.stdin:
                rdp_process.stdin.write("q\n");rdp_process.stdin.flush()
            try:rdp_code=rdp_process.wait(timeout=45);rdp_timeout=False
            except subprocess.TimeoutExpired:rdp_code=None;rdp_timeout=True
    valid_profile=False
    if profile.is_file() and profile.stat().st_size>=4096:
        with profile.open("rb") as stream:valid_profile=stream.read(8)==b"AMD_RDF "
    checks=target_code==0 and rdp_code==0 and not target_timeout and not rdp_timeout and valid_profile and ready is not False
    manifest={"schema":1,"checks_pass":checks,"target_command":command,"target_returncode":target_code,
              "target_host_timeout":target_timeout,"target_pid":None if target is None else target.pid,
              "rdp_command":rdp_command,"rdp_returncode":rdp_code,"rdp_timeout":rdp_timeout,
              "rdp_pid":None if rdp_process is None else rdp_process.pid,"capture_mode":"dispatch",
              "counter_collection":True,"dispatch_start":args.dispatch_start,"dispatch_count":args.dispatch_count,
              "delay_ms":None if args.ready_token else args.delay_ms,"ready_token":args.ready_token or None,
              "ready_token_observed":ready,"trigger_mode":"interactive_after_ready" if args.ready_token else "fixed_delay_auto",
              "profile_valid_amd_rdf":valid_profile,
              "profile_bytes":profile.stat().st_size if profile.exists() else 0,
              "profile_sha256":sha256(profile) if valid_profile else None,
              "persistent_clock_change_requested":False,"tdr_change_requested":False,"automatic_retry":False,
              "wall_seconds":time.monotonic()-started}
    with (args.output/"manifest.json").open("x",encoding="utf-8") as stream:json.dump(manifest,stream,indent=2)
    print(json.dumps(manifest,indent=2),flush=True)
    return 0 if checks else 2


if __name__=="__main__":raise SystemExit(main())
