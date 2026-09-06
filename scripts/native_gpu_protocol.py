"""Bounded metadata-only protocol. No image bytes, pickle, implicit resize or retry."""
import json

MAX_MESSAGE_BYTES=8192


def worker_identity(launcher,actual,observed_parent,launcher_running):
    if type(actual) is not int or actual<=0 or not launcher_running:
        raise ValueError('invalid/exited worker identity')
    if actual!=launcher and observed_parent!=launcher:
        raise ValueError('worker must be launched process or its verified direct child')
    return actual


def parse(line):
    if not isinstance(line,str) or len(line.encode('utf-8'))>MAX_MESSAGE_BYTES:
        raise ValueError('metadata message too large')
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('duplicate metadata field')
            out[k]=v
        return out
    result=json.loads(line,object_pairs_hook=unique)
    if not isinstance(result,dict):raise ValueError('metadata object required')
    return result


def descriptor(fields,producer,worker,*,max_width=768,max_height=512):
    if type(max_width) is not int or type(max_height) is not int or not 1<=max_width<=3840 or not 1<=max_height<=2160:
        raise ValueError('explicit bounded geometry policy required')
    if (not isinstance(fields,list) or len(fields)!=16
            or any(type(v) is not int or not 0<=v<2**64 for v in fields)):
        raise ValueError('sixteen uint64 metadata fields required')
    v,pid,target,generation,w,h,pitch,span,heap,luid,*rest=fields
    if (v!=1 or pid!=producer or target!=worker or not pid or pid==target or not generation
            or not 1<=w<=max_width or not 1<=h<=max_height or pitch%256 or pitch<w*8
            or span<(h-1)*pitch+w*8 or heap<span or heap%65536 or heap>128*1024*1024
            or not luid or not all(rest[:4]) or len(set(rest[:4]))!=4 or rest[4:]!=[0,0]):
        raise ValueError('unsupported descriptor ownership/geometry/budget')
    return w,h,generation


class Commands:
    def __init__(self,generation):self.generation=generation;self.sequence=0;self.state='READY'
    def frame(self,message):
        if set(message)!={'op','generation','sequence','seed'} or message['op']!='frame':raise ValueError('frame metadata only')
        if (self.state!='READY' or message['generation']!=self.generation
                or type(message['sequence']) is not int or message['sequence']!=self.sequence+1
                or type(message['seed']) is not int or not 0<=message['seed']<=0xffffffff):
            raise ValueError('pending/stale generation, frame or seed')
        self.sequence=message['sequence'];self.state='AWAITING_CONSUMPTION'
    def consumed(self,message):
        if (set(message)!={'op','sequence','generation'} or message['op']!='consumed'
                or self.state!='AWAITING_CONSUMPTION' or message['sequence']!=self.sequence or message['generation']!=self.generation):
            raise ValueError('missing or stale consumer acknowledgement')
        self.state='READY'
    def close(self,message):
        if message!={'op':'close'} or self.state!='READY':raise ValueError('cannot close pending resource lease')
        self.state='CLOSED'
