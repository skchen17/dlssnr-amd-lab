#pragma once

#include <cstdint>

// Lab-only diagnostics ABI. This is not part of NVAPI and is never returned by
// nvapi_QueryInterface; it exists solely so the self-test can prove that the
// drop-in surface performed real HIP work.
struct NvapiAmdDiagnosticsV1 {
    uint32_t struct_size;
    uint32_t schema;
    uint32_t module_creates;
    uint32_t function_creates;
    uint32_t launch_chain_calls;
    uint32_t kernels_submitted;
    uint32_t gpu_markers;
    uint32_t validation_mismatches;
    uint32_t function_destroys;
    uint32_t module_destroys;
    uint32_t live_functions;
    uint32_t live_modules;
    uint32_t hip_device_count;
    uint32_t resource_registrations;
    uint32_t resource_unregistrations;
    uint32_t live_resources;
    uint32_t address_translations;
    uint32_t translation_failures;
    uint32_t boundary_kernels;
    uint32_t counts_as_s6;
    uint32_t neural_math_executed;
    uint32_t merged_texture_sampler_calls;
    uint32_t independent_descriptor_calls;
    uint32_t live_descriptor_objects;
    uint32_t d3d12_copy_dispatches;
    uint32_t pipeline_failures;
    char hip_device[128];
    char hip_arch[64];
    char classification[64];
};

using NvapiAmdGetDiagnostics_t = int(__cdecl*)(NvapiAmdDiagnosticsV1*);
using NvapiAmdResetDiagnostics_t = int(__cdecl*)();
using NvapiAmdRegisterExternalBuffer_t = int(__cdecl*)(void* d3d12_resource,
                                                       void* shared_handle,
                                                       uint64_t byte_size);
using NvapiAmdUnregisterExternalBuffer_t = int(__cdecl*)(void* d3d12_resource);
using NvapiAmdRegisterDescriptorResource_t = int(__cdecl*)(uint64_t descriptor_cpu,
                                                           void* d3d12_resource);
