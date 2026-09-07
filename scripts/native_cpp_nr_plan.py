"""C++-owned HIP Graph runner; Python participates only during initialization.

This bridge captures the currently selected, already-correct kernel topology.
It removes Python/PyTorch operation submission from the per-frame hot path but
does not claim that all captured ATen/library kernels have been reauthored.
"""
from __future__ import annotations

import ctypes as ct
import threading
from pathlib import Path


class Desc(ct.Structure):
    _fields_=(('workspace_bytes',ct.c_uint64),('weight_bytes',ct.c_uint64),
              ('max_width',ct.c_uint32),('max_height',ct.c_uint32))


class Bindings(ct.Structure):
    _fields_=(('struct_size',ct.c_uint32),('abi_version',ct.c_uint32),
              ('input',ct.c_void_p),('output',ct.c_void_p),
              ('history',ct.c_void_p),('next_history',ct.c_void_p),
              ('motion',ct.c_void_p),('depth',ct.c_void_p),
              ('exposure',ct.c_void_p),('controls',ct.c_void_p),
              ('width',ct.c_uint32),('height',ct.c_uint32),
              ('frame_id',ct.c_uint64),('resource_generation',ct.c_uint64),
              ('reset',ct.c_uint32),('flags',ct.c_uint32))


class ShapeDescV3(ct.Structure):
    _fields_=(('struct_size',ct.c_uint32),('render_width',ct.c_uint32),
              ('render_height',ct.c_uint32),('output_width',ct.c_uint32),
              ('output_height',ct.c_uint32),('valid_left',ct.c_uint32),
              ('valid_top',ct.c_uint32),('valid_width',ct.c_uint32),
              ('valid_height',ct.c_uint32),('color_mode',ct.c_uint32),
              ('color_format',ct.c_uint32),('motion_format',ct.c_uint32),
              ('depth_format',ct.c_uint32))


class BindingsV3(ct.Structure):
    _fields_=(('struct_size',ct.c_uint32),('abi_version',ct.c_uint32),
              ('current_color',ct.c_void_p),('output_residual',ct.c_void_p),
              ('history',ct.c_void_p),('next_history',ct.c_void_p),
              ('motion',ct.c_void_p),('depth',ct.c_void_p),
              ('exposure',ct.c_void_p),('controls',ct.c_void_p),
              ('render_width',ct.c_uint32),('render_height',ct.c_uint32),
              ('output_width',ct.c_uint32),('output_height',ct.c_uint32),
              ('valid_left',ct.c_uint32),('valid_top',ct.c_uint32),
              ('valid_width',ct.c_uint32),('valid_height',ct.c_uint32),
              ('jitter_x',ct.c_float),('jitter_y',ct.c_float),
              ('motion_scale_x',ct.c_float),('motion_scale_y',ct.c_float),
              ('pre_exposure',ct.c_float),('exposure_scale',ct.c_float),
              ('frame_id',ct.c_uint64),('resource_generation',ct.c_uint64),
              ('reset',ct.c_uint32),('flags',ct.c_uint32))


class ExternalSyncV3(ct.Structure):
    _fields_=(('struct_size',ct.c_uint32),('reserved',ct.c_uint32),
              ('input_ready',ct.c_void_p),('input_value',ct.c_uint64),
              ('output_done',ct.c_void_p),('output_value',ct.c_uint64))


class PerformanceStats(ct.Structure):
    _fields_=(('submitted_frames',ct.c_uint64),('completed_frames',ct.c_uint64),
              ('last_gpu_ms',ct.c_double),('total_gpu_ms',ct.c_double),
              ('min_gpu_ms',ct.c_double),('max_gpu_ms',ct.c_double))


class ModelPackageStats(ct.Structure):
    _fields_=(('package_version',ct.c_uint32),('precision_profile',ct.c_uint32),
              ('selected_weight_bytes',ct.c_uint64),
              ('selected_weight_segments',ct.c_uint32),
              ('temporal_contract_verified',ct.c_uint32),
              ('architecture',ct.c_char*32),('temporal_contract_id',ct.c_char*64),
              ('target_arch',ct.c_char*16),('original_model_sha256',ct.c_char*65),
              ('selected_weights_sha256',ct.c_char*65))


def validate_v3_layouts():
    """Catch accidental Python/C ABI drift before any GPU resource is touched."""
    expected = {ShapeDescV3: 52, BindingsV3: 152, ExternalSyncV3: 40,
                ApproxStageBlock: 240, ApproxSplit512Block: 248,
                ApproxVitBlock: 152, ApproxBottleneck: 112,
                ApproxScaleTransition: 96,
                PerformanceStats: 48, ModelPackageStats: 272}
    actual = {kind.__name__: ct.sizeof(kind) for kind in expected}
    wrong = {kind.__name__: (ct.sizeof(kind), size)
             for kind, size in expected.items() if ct.sizeof(kind) != size}
    if wrong:
        raise RuntimeError(f'NRPlan ABI v3 layout mismatch: {wrong}')
    return actual


class GraphStats(ct.Structure):
    _fields_=(('total_nodes',ct.c_uint64),('kernel_nodes',ct.c_uint64),('memcpy_nodes',ct.c_uint64))


class KernelNodeInfo(ct.Structure):
    _fields_=(('graph_node_ordinal',ct.c_uint64),('kernel_ordinal',ct.c_uint64),
              ('grid_x',ct.c_uint32),('grid_y',ct.c_uint32),('grid_z',ct.c_uint32),
              ('block_x',ct.c_uint32),('block_y',ct.c_uint32),('block_z',ct.c_uint32),
              ('dynamic_shared_bytes',ct.c_uint32),('registers_per_thread',ct.c_uint32),
              ('static_shared_bytes',ct.c_uint64),('local_bytes_per_thread',ct.c_uint64),
              ('max_threads_per_block',ct.c_uint32),('name',ct.c_char*512))


class ResourceStats(ct.Structure):
    _fields_=(('workspace_bytes',ct.c_uint64),('weight_bytes',ct.c_uint64),
              ('arena_region_count',ct.c_uint64),('owns_workspace',ct.c_uint8),
              ('owns_weights',ct.c_uint8),('owns_graph_source',ct.c_uint8),
              ('owns_graph_executable',ct.c_uint8),('graph_references_external_allocations',ct.c_uint8),
              ('deployment_ready',ct.c_uint8),('temporal_contract_verified',ct.c_uint8),
              ('precision_profile',ct.c_uint8),
              ('frame_bindings_abi_version',ct.c_uint32),
              ('native_stage_block_count',ct.c_uint32),
              ('complete_native_topology',ct.c_uint8),('reserved',ct.c_uint8*3))


class ApproxStageBlock(ct.Structure):
    _fields_=(('struct_size',ct.c_uint32),('channels',ct.c_uint32),
              ('windows',ct.c_uint32),('record_number',ct.c_uint32),
              ('feature_width',ct.c_uint32),('feature_height',ct.c_uint32),
              ('origin_x',ct.c_int32),('origin_y',ct.c_int32),
              ('raw_resident_offset',ct.c_uint64),('grouped_offset',ct.c_uint64),
              ('seed_fp16_offset',ct.c_uint64),('post_fp16_offset',ct.c_uint64),
              ('post_resident_offset',ct.c_uint64),
              ('qkv_projection_fp16_offset',ct.c_uint64),
              ('q_offset',ct.c_uint64),('k_offset',ct.c_uint64),
              ('v_offset',ct.c_uint64),('value_offset',ct.c_uint64),
              ('output_fp16_offset',ct.c_uint64),('next_resident_offset',ct.c_uint64),
              ('ffn_expand_weight_offset',ct.c_uint64),
              ('ffn_contract_weight_offset',ct.c_uint64),
              ('ffn_mix_weight_offset',ct.c_uint64),('ffn_scale_weight_offset',ct.c_uint64),
              ('a_index_weight_offset',ct.c_uint64),
              ('residual_index_weight_offset',ct.c_uint64),
              ('ffn_permutation_weight_offset',ct.c_uint64),
              ('ffn_inverse_permutation_weight_offset',ct.c_uint64),
              ('qkv_weight_offset',ct.c_uint64),('qscale_weight_offset',ct.c_uint64),
              ('permutation_weight_offset',ct.c_uint64),
              ('position_bias_weight_offset',ct.c_uint64),
              ('project_weight_offset',ct.c_uint64),
              ('residual_scale_weight_offset',ct.c_uint64))


class ApproxSplit512Block(ct.Structure):
    _fields_=(('struct_size',ct.c_uint32),('windows',ct.c_uint32),
              ('feature_width',ct.c_uint32),('feature_height',ct.c_uint32),
              ('origin_x',ct.c_int32),('origin_y',ct.c_int32),
              ('record_number',ct.c_uint32),('reserved',ct.c_uint32),
              ('raw_resident_offset',ct.c_uint64),('projected_offset',ct.c_uint64),
              ('grouped_offset',ct.c_uint64),('post_fp16_offset',ct.c_uint64),
              ('post_resident_offset',ct.c_uint64),('q_offset',ct.c_uint64),
              ('k_offset',ct.c_uint64),('v_offset',ct.c_uint64),
              ('value_offset',ct.c_uint64),('output_fp16_offset',ct.c_uint64),
              ('next_resident_offset',ct.c_uint64),
              ('preproject_weight_offset',ct.c_uint64),
              ('expand_weight_offset',ct.c_uint64),
              ('contract_weight_offset',ct.c_uint64),
              ('ffn_project_weight_offset',ct.c_uint64),
              ('ffn_scale_weight_offset',ct.c_uint64),
              ('a_index_weight_offset',ct.c_uint64),
              ('residual_index_weight_offset',ct.c_uint64),
              ('perm64_weight_offset',ct.c_uint64),
              ('perm256_weight_offset',ct.c_uint64),
              ('ffn_permutation_weight_offset',ct.c_uint64),
              ('qkv_weight_offset',ct.c_uint64),('qscale_weight_offset',ct.c_uint64),
              ('attention_permutation_weight_offset',ct.c_uint64),
              ('position_bias_weight_offset',ct.c_uint64),
              ('attention_project_weight_offset',ct.c_uint64),
              ('attention_scale_weight_offset',ct.c_uint64))


class ApproxVitBlock(ct.Structure):
    _fields_=(('struct_size',ct.c_uint32),('tokens',ct.c_uint32),
              ('record_number',ct.c_uint32),('reserved',ct.c_uint32),
              ('input_offset',ct.c_uint64),('hidden_offset',ct.c_uint64),
              ('post_offset',ct.c_uint64),('q_offset',ct.c_uint64),
              ('k_offset',ct.c_uint64),('v_offset',ct.c_uint64),
              ('value_offset',ct.c_uint64),('next_offset',ct.c_uint64),
              ('expand_weight_offset',ct.c_uint64),
              ('contract_weight_offset',ct.c_uint64),
              ('qkv_weight_offset',ct.c_uint64),
              ('q_scale_weight_offset',ct.c_uint64),
              ('project_weight_offset',ct.c_uint64),
              ('ffn_scale_weight_offset',ct.c_uint64),
              ('attention_scale_weight_offset',ct.c_uint64),
              ('perm1024_weight_offset',ct.c_uint64),
              ('perm4096_weight_offset',ct.c_uint64))


class ApproxBottleneck(ct.Structure):
    _fields_=(('struct_size',ct.c_uint32),('tokens',ct.c_uint32),
              ('feature_width',ct.c_uint32),('feature_height',ct.c_uint32),
              ('low_width',ct.c_uint32),('low_height',ct.c_uint32),
              ('reserved0',ct.c_uint32),('reserved1',ct.c_uint32),
              ('c512_input_offset',ct.c_uint64),('c512_skip_offset',ct.c_uint64),
              ('vit_input_offset',ct.c_uint64),('vit_output_offset',ct.c_uint64),
              ('c512_output_offset',ct.c_uint64),
              ('encoder_weight_offset',ct.c_uint64),
              ('encoder_permutation_offset',ct.c_uint64),
              ('decoder_weight_offset',ct.c_uint64),
              ('decoder_scale_offset',ct.c_uint64),
              ('decoder_permutation_offset',ct.c_uint64))


class ApproxScaleTransition(ct.Structure):
    _fields_=(('struct_size',ct.c_uint32),('direction',ct.c_uint32),
              ('anchor_record',ct.c_uint32),('channels',ct.c_uint32),
              ('source_width',ct.c_uint32),('source_height',ct.c_uint32),
              ('target_width',ct.c_uint32),('target_height',ct.c_uint32),
              ('source_origin_x',ct.c_int32),('source_origin_y',ct.c_int32),
              ('source_offset',ct.c_uint64),('source_resident_offset',ct.c_uint64),
              ('skip_offset',ct.c_uint64),
              ('target_offset',ct.c_uint64),('project_weight_offset',ct.c_uint64),
              ('permutation_weight_offset',ct.c_uint64),
              ('skip_scale_weight_offset',ct.c_uint64))


class ArenaRegion(ct.Structure):
    _fields_=(('offset',ct.c_uint64),('bytes',ct.c_uint64),('alignment',ct.c_uint32),
              ('kind',ct.c_uint32),('name',ct.c_char*64))


def approximate_stage_descriptor_array(topology):
    if topology.get('abi_version') != 3 or \
            topology.get('status') != 'PARTIAL_NATIVE_SWIN32_256_BLOCKS_NOT_RUNTIME_ACCEPTED' or \
            topology.get('complete_native_topology') is not False:
        raise ValueError('unrecognized partial native-stage topology')
    fields = [name for name, _ in ApproxStageBlock._fields_
              if name not in ('struct_size', 'channels', 'windows', 'record_number',
                              'feature_width', 'feature_height', 'origin_x', 'origin_y')]
    values = []
    for row in topology.get('blocks', []):
        if row.get('struct_size') != ct.sizeof(ApproxStageBlock):
            raise ValueError('native-stage descriptor ABI mismatch')
        values.append(ApproxStageBlock(ct.sizeof(ApproxStageBlock), row['channels'],
                                       row['windows'], row['record_number'], row['feature_width'],
                                       row['feature_height'], row['origin_x'], row['origin_y'],
                                       *(row[name] for name in fields)))
    if len(values) != topology.get('block_count') or not values:
        raise ValueError('native-stage block count mismatch')
    return (ApproxStageBlock * len(values))(*values)


def approximate_split512_descriptor_array(topology):
    if topology.get('abi_version') != 3 or \
            topology.get('status') != 'PARTIAL_NATIVE_C512_SPLIT_NOT_RUNTIME_ACCEPTED' or \
            topology.get('complete_native_topology') is not False:
        raise ValueError('unrecognized partial native C512 topology')
    header = {'struct_size', 'windows', 'feature_width', 'feature_height',
              'origin_x', 'origin_y', 'record_number', 'reserved'}
    fields = [name for name, _ in ApproxSplit512Block._fields_ if name not in header]
    values = []
    for row in topology.get('blocks', []):
        if row.get('struct_size') != ct.sizeof(ApproxSplit512Block):
            raise ValueError('native C512 descriptor ABI mismatch')
        values.append(ApproxSplit512Block(ct.sizeof(ApproxSplit512Block), row['windows'],
            row['feature_width'], row['feature_height'], row['origin_x'], row['origin_y'],
            row['record_number'], 0, *(row[name] for name in fields)))
    if len(values) != topology.get('block_count') or not values:
        raise ValueError('native C512 block count mismatch')
    return (ApproxSplit512Block * len(values))(*values)


def approximate_vit_descriptor_array(topology):
    if topology.get('abi_version') != 3 or \
            topology.get('status') != 'PARTIAL_NATIVE_VIT8_NOT_RUNTIME_ACCEPTED' or \
            topology.get('complete_native_topology') is not False:
        raise ValueError('unrecognized partial native ViT topology')
    header = {'struct_size', 'tokens', 'record_number', 'reserved'}
    fields = [name for name, _ in ApproxVitBlock._fields_ if name not in header]
    values = []
    for row in topology.get('blocks', []):
        if row.get('struct_size') != ct.sizeof(ApproxVitBlock):
            raise ValueError('native ViT descriptor ABI mismatch')
        values.append(ApproxVitBlock(ct.sizeof(ApproxVitBlock), row['tokens'],
                                     row['record_number'], 0,
                                     *(row[name] for name in fields)))
    if len(values) != 8 or len(values) != topology.get('block_count'):
        raise ValueError('native ViT block count mismatch')
    return (ApproxVitBlock * len(values))(*values)


def approximate_bottleneck_descriptor(topology):
    if topology.get('abi_version') != 3 or \
            topology.get('status') != 'PARTIAL_NATIVE_BOTTLENECK_TRANSITIONS_NOT_RUNTIME_ACCEPTED' or \
            topology.get('complete_native_topology') is not False:
        raise ValueError('unrecognized native bottleneck topology')
    row = topology.get('descriptor', {})
    if row.get('struct_size') != ct.sizeof(ApproxBottleneck):
        raise ValueError('native bottleneck descriptor ABI mismatch')
    header = {'struct_size', 'tokens', 'feature_width', 'feature_height',
              'low_width', 'low_height', 'reserved0', 'reserved1'}
    fields = [name for name, _ in ApproxBottleneck._fields_ if name not in header]
    return ApproxBottleneck(ct.sizeof(ApproxBottleneck), row['tokens'],
                            row['feature_width'], row['feature_height'],
                            row['low_width'], row['low_height'], 0, 0,
                            *(row[name] for name in fields))


def approximate_scale_transition_descriptor_array(topology):
    if topology.get('abi_version') != 3 or \
            topology.get('status') != 'PARTIAL_NATIVE_SCALE_TRANSITIONS_NOT_RUNTIME_ACCEPTED' or \
            topology.get('complete_native_topology') is not False:
        raise ValueError('unrecognized native scale-transition topology')
    header = {'struct_size', 'direction', 'anchor_record', 'channels',
              'source_width', 'source_height', 'target_width', 'target_height',
              'source_origin_x', 'source_origin_y'}
    fields = [name for name, _ in ApproxScaleTransition._fields_ if name not in header]
    values = []
    for row in topology.get('descriptors', []):
        if row.get('struct_size') != ct.sizeof(ApproxScaleTransition):
            raise ValueError('native transition descriptor ABI mismatch')
        values.append(ApproxScaleTransition(ct.sizeof(ApproxScaleTransition),
            row['direction'], row['anchor_record'], row['channels'],
            row['source_width'], row['source_height'], row['target_width'],
            row['target_height'], row['source_origin_x'], row['source_origin_y'],
            *(row[name] for name in fields)))
    if len(values) != 8 or len(values) != topology.get('transition_count'):
        raise ValueError('native transition descriptor count mismatch')
    return (ApproxScaleTransition * len(values))(*values)


class CapturedNRPlan:
    classification='CPP_OWNED_CAPTURED_KERNEL_GRAPH'
    full_math_reauthored=False
    python_in_hot_path=False

    def __init__(self,dll,forward,example,*,wait,max_width=3840,max_height=2160,lifetimes=(),
                 weight_blob=b'\0',workspace_bytes=1,arena_regions=(),stage_iterator=None):
        import torch
        if not torch.version.hip or not example.is_cuda or example.dtype!=torch.float16 or example.ndim!=3:
            raise ValueError('contiguous ROCm FP16 HWC input required')
        if not example.is_contiguous():raise ValueError('contiguous input required')
        self.owner=threading.get_ident();self.closed=False;self.sequence=0
        self.library=ct.CDLL(str(Path(dll).resolve()));self.lifetimes=tuple(lifetimes)
        self._bind_abi()
        if not isinstance(weight_blob,(bytes,bytearray,memoryview)) or not len(weight_blob):
            raise ValueError('non-empty immutable model weight blob required')
        if type(workspace_bytes) is not int or workspace_bytes<=0:raise ValueError('positive workspace size required')
        weight_blob=bytes(weight_blob)
        self.handle=ct.c_void_p();desc=Desc(workspace_bytes,len(weight_blob),max_width,max_height)
        self._call(self.library.nrPlanCreate(ct.byref(desc),ct.byref(self.handle)),'create')
        upload=(ct.c_ubyte*len(weight_blob)).from_buffer_copy(weight_blob)
        self._call(self.library.nrPlanUploadWeights(self.handle,upload,len(weight_blob)),'upload weights')
        regions=[]
        for region in arena_regions:
            name=str(region['name']).encode('utf-8')
            if not name or len(name)>=64:raise ValueError('arena region name must be 1..63 UTF-8 bytes')
            regions.append(ArenaRegion(region['offset'],region['bytes'],region.get('alignment',256),
                                       region.get('kind',0),name))
        if regions:
            values=(ArenaRegion*len(regions))(*regions)
            self._call(self.library.nrPlanConfigureArena(self.handle,values,len(values)),'configure arena')
        stream_ptr=ct.c_void_p();self._call(self.library.nrPlanGetStream(self.handle,ct.byref(stream_ptr)),'get stream')
        self.stream_pointer=stream_ptr.value;self.stream=torch.cuda.ExternalStream(self.stream_pointer)
        self.input=example.clone();self.output=None
        # Warm handles, workspaces and allocator state on the exact capture stream.
        with torch.no_grad(),torch.cuda.stream(self.stream):warm=forward(self.input)
        wait(self.stream);del warm
        # PyTorch owns the capture pool during initialization so every ATen
        # temporary has a stable address. NRPlan clones/instantiates the raw
        # graph and is the only per-frame submitter after this point.
        self.capture_graph=torch.cuda.CUDAGraph(keep_graph=True)
        self.stage_marker_scratch=torch.empty(1,device=example.device,dtype=torch.int32) if stage_iterator else None
        with torch.no_grad(),torch.cuda.stream(self.stream):
            self.capture_graph.capture_begin()
            try:
                if stage_iterator is None:self.output=forward(self.input)
                else:
                    self._record_stage_marker(0)
                    expected=0
                    for block,value in stage_iterator(self.input):
                        if block!=expected:raise RuntimeError('census stage order mismatch')
                        self._record_stage_marker(block+1);expected+=1
                    if expected!=71:raise RuntimeError('census did not cover all 71 blocks')
                    self.output=value
            except BaseException:
                # End an invalidated capture before destroying the owning HIP
                # stream. The original exception is the useful diagnostic.
                try:self.capture_graph.capture_end()
                except BaseException:pass
                self.close_noexcept();raise
            else:self.capture_graph.capture_end()
        self._call(self.library.nrPlanAdoptGraph(self.handle,ct.c_void_p(self.capture_graph.raw_cuda_graph())),'adopt graph')
        if self.output is None or not self.output.is_cuda or not self.output.is_contiguous():
            self.close_noexcept();raise RuntimeError('capture did not produce contiguous GPU output')
        stats=GraphStats();self._call(self.library.nrPlanGetGraphStats(self.handle,ct.byref(stats)),'get graph stats')
        self.graph_stats={'total_nodes':stats.total_nodes,'kernel_nodes':stats.kernel_nodes,'memcpy_nodes':stats.memcpy_nodes}
        self.resource_stats=self._resource_stats()
        self.target=torch.empty_like(self.output)
        self._call(self.library.nrPlanSetStaticIO(self.handle,ct.c_void_p(self.input.data_ptr()),self.input.numel()*self.input.element_size(),
            ct.c_void_p(self.output.data_ptr()),self.output.numel()*self.output.element_size(),example.shape[1],example.shape[0]),'set static IO')

    def _bind_abi(self):
        d=self.library
        d.nrPlanCreate.argtypes=[ct.POINTER(Desc),ct.POINTER(ct.c_void_p)]
        d.nrPlanGetStream.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p)]
        d.nrPlanAdoptGraph.argtypes=[ct.c_void_p,ct.c_void_p]
        d.nrPlanGetGraphStats.argtypes=[ct.c_void_p,ct.POINTER(GraphStats)]
        d.nrPlanGetKernelNodeInfos.argtypes=[ct.c_void_p,ct.POINTER(KernelNodeInfo),ct.c_uint64,ct.POINTER(ct.c_uint64)]
        d.nrPlanDebugDotPrint.argtypes=[ct.c_void_p,ct.c_char_p]
        d.nrPlanRecordStageMarker.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_void_p]
        d.nrPlanGetResourceStats.argtypes=[ct.c_void_p,ct.POINTER(ResourceStats)]
        d.nrPlanConfigureArena.argtypes=[ct.c_void_p,ct.POINTER(ArenaRegion),ct.c_uint64]
        d.nrPlanConfigureApproxStageBlocks.argtypes=[ct.c_void_p,ct.POINTER(ApproxStageBlock),
                                                     ct.c_uint32,ct.c_uint8]
        d.nrPlanConfigureApproxSplit512Blocks.argtypes=[ct.c_void_p,
            ct.POINTER(ApproxSplit512Block),ct.c_uint32]
        d.nrPlanConfigureApproxVitBlocks.argtypes=[ct.c_void_p,
            ct.POINTER(ApproxVitBlock),ct.c_uint32]
        d.nrPlanConfigureApproxBottleneck.argtypes=[ct.c_void_p,
            ct.POINTER(ApproxBottleneck)]
        d.nrPlanConfigureApproxScaleTransitions.argtypes=[ct.c_void_p,
            ct.POINTER(ApproxScaleTransition),ct.c_uint32]
        d.nrPlanInitializeArenaFromDevice.argtypes=[ct.c_void_p,ct.c_uint64,ct.c_void_p,
                                                    ct.c_uint64]
        d.nrPlanDebugCopyArenaToDevice.argtypes=[ct.c_void_p,ct.c_uint64,ct.c_void_p,
                                                 ct.c_uint64]
        d.nrPlanUploadWeights.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64]
        d.nrPlanSetRecorder.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p]
        d.nrPlanSetRecorderV3.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p]
        d.nrPlanSetPrecisionProfile.argtypes=[ct.c_void_p,ct.c_uint32]
        d.nrPlanPrepareShape.argtypes=[ct.c_void_p,ct.POINTER(ShapeDescV3)]
        d.nrPlanLoadModelPackage.argtypes=[ct.c_void_p,ct.c_char_p,ct.POINTER(ModelPackageStats)]
        d.nrPlanFinalize.argtypes=[ct.c_void_p]
        d.nrPlanSetStaticIO.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64,ct.c_void_p,ct.c_uint64,ct.c_uint32,ct.c_uint32]
        d.nrPlanSubmit.argtypes=[ct.c_void_p,ct.POINTER(Bindings),ct.c_void_p,ct.c_void_p]
        d.nrPlanSubmitV3.argtypes=[ct.c_void_p,ct.POINTER(BindingsV3),ct.c_void_p,ct.c_void_p]
        d.nrPlanSubmitExternalV3.argtypes=[ct.c_void_p,ct.POINTER(BindingsV3),
                                          ct.POINTER(ExternalSyncV3)]
        d.nrPlanGetPerformanceStats.argtypes=[ct.c_void_p,ct.POINTER(PerformanceStats)]
        d.nrPlanDestroy.argtypes=[ct.c_void_p]
        for name in ('nrPlanCreate','nrPlanGetStream','nrPlanAdoptGraph','nrPlanGetGraphStats',
                     'nrPlanGetKernelNodeInfos','nrPlanDebugDotPrint','nrPlanRecordStageMarker',
                     'nrPlanGetResourceStats','nrPlanConfigureArena','nrPlanConfigureApproxStageBlocks',
                     'nrPlanConfigureApproxSplit512Blocks','nrPlanConfigureApproxVitBlocks',
                     'nrPlanConfigureApproxBottleneck','nrPlanConfigureApproxScaleTransitions',
                     'nrPlanInitializeArenaFromDevice','nrPlanDebugCopyArenaToDevice',
                     'nrPlanUploadWeights',
                     'nrPlanSetRecorder','nrPlanSetRecorderV3','nrPlanSetPrecisionProfile',
                     'nrPlanPrepareShape','nrPlanLoadModelPackage','nrPlanFinalize',
                     'nrPlanSetStaticIO','nrPlanSubmit','nrPlanSubmitV3',
                     'nrPlanSubmitExternalV3',
                     'nrPlanGetPerformanceStats','nrPlanDestroy'):
            getattr(d,name).restype=ct.c_int

    def _resource_stats(self):
        value=ResourceStats();self._call(self.library.nrPlanGetResourceStats(self.handle,ct.byref(value)),'get resource stats')
        return {'workspace_bytes':value.workspace_bytes,'weight_bytes':value.weight_bytes,
                'arena_region_count':value.arena_region_count,'owns_workspace':bool(value.owns_workspace),
                'owns_weights':bool(value.owns_weights),'owns_graph_source':bool(value.owns_graph_source),
                'owns_graph_executable':bool(value.owns_graph_executable),
                'graph_references_external_allocations':bool(value.graph_references_external_allocations),
                'deployment_ready':bool(value.deployment_ready),
                'temporal_contract_verified':bool(value.temporal_contract_verified),
                'precision_profile':value.precision_profile,
                'frame_bindings_abi_version':value.frame_bindings_abi_version,
                'native_stage_block_count':value.native_stage_block_count,
                'complete_native_topology':bool(value.complete_native_topology)}

    def kernel_node_census(self):
        count=ct.c_uint64()
        self._call(self.library.nrPlanGetKernelNodeInfos(self.handle,None,0,ct.byref(count)),'count kernel nodes')
        values=(KernelNodeInfo*count.value)()
        self._call(self.library.nrPlanGetKernelNodeInfos(self.handle,values,count.value,ct.byref(count)),'read kernel nodes')
        return [{'graph_node_ordinal':v.graph_node_ordinal,'kernel_ordinal':v.kernel_ordinal,
                 'name':bytes(v.name).split(b'\0',1)[0].decode('utf-8','replace'),
                 'grid':[v.grid_x,v.grid_y,v.grid_z],'block':[v.block_x,v.block_y,v.block_z],
                 'dynamic_shared_bytes':v.dynamic_shared_bytes,
                 'registers_per_thread':v.registers_per_thread,
                 'static_shared_bytes':v.static_shared_bytes,
                 'local_bytes_per_thread':v.local_bytes_per_thread,
                 'max_threads_per_block':v.max_threads_per_block} for v in values]

    def write_debug_dot(self,path):
        path=Path(path).resolve()
        if path.exists():raise FileExistsError(path)
        self._call(self.library.nrPlanDebugDotPrint(self.handle,str(path).encode()),'write graph DOT')
        return path

    def _record_stage_marker(self,boundary):
        self._call(self.library.nrPlanRecordStageMarker(ct.c_void_p(self.stream_pointer),boundary,
                   ct.c_void_p(self.stage_marker_scratch.data_ptr())),'record stage marker')

    @staticmethod
    def _call(code,operation):
        if code:raise RuntimeError(f'NRPlan {operation} failed: HIP error {code}')

    def submit(self,source,target=None,*,resource_generation=1,history=None,next_history=None,
               motion=None,depth=None,exposure=None,controls=None,reset=False,flags=0):
        import torch
        if self.closed or threading.get_ident()!=self.owner:raise RuntimeError('closed plan or changed owner')
        if source.shape!=self.input.shape or source.dtype!=self.input.dtype or source.device!=self.input.device or not source.is_contiguous():
            raise ValueError('input geometry/dtype/device changed; build another plan')
        target=self.target if target is None else target
        if target.shape!=self.output.shape or target.dtype!=self.output.dtype or target.device!=self.output.device or not target.is_contiguous():
            raise ValueError('output target mismatch')
        def pointer(value):return value.data_ptr() if value is not None else 0
        h,w,_=source.shape;binding=Bindings(ct.sizeof(Bindings),2,source.data_ptr(),target.data_ptr(),
            pointer(history),pointer(next_history),pointer(motion),pointer(depth),pointer(exposure),
            pointer(controls),w,h,self.sequence,resource_generation,int(bool(reset)),flags)
        self._call(self.library.nrPlanSubmit(self.handle,ct.byref(binding),None,None),'submit')
        self.sequence+=1;return target

    def close(self,wait):
        if self.closed:return
        wait(self.stream);self._call(self.library.nrPlanDestroy(self.handle),'destroy');self.closed=True

    def close_noexcept(self):
        if not getattr(self,'closed',True) and getattr(self,'handle',None):
            try:self.library.nrPlanDestroy(self.handle)
            finally:self.closed=True

    def __del__(self):
        self.close_noexcept()
