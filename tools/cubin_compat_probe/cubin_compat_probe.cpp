#include <cuda.h>

#include <cstdio>

static const char* ErrorName(CUresult result) {
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    return name ? name : "CUDA_ERROR_UNKNOWN";
}

static const char* ErrorText(CUresult result) {
    const char* text = nullptr;
    cuGetErrorString(result, &text);
    return text ? text : "unknown CUDA driver error";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <cubin> [cubin ...]\n", argv[0]);
        return 2;
    }

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuInit: %s (%d): %s\n", ErrorName(result),
                     static_cast<int>(result), ErrorText(result));
        return 3;
    }

    int device_count = 0;
    result = cuDeviceGetCount(&device_count);
    if (result != CUDA_SUCCESS || device_count < 1) {
        std::fprintf(stderr, "cuDeviceGetCount: %s (%d): %s; count=%d\n",
                     ErrorName(result), static_cast<int>(result), ErrorText(result),
                     device_count);
        return 4;
    }

    CUdevice device = 0;
    cuDeviceGet(&device, 0);
    char device_name[256] = {};
    int major = 0;
    int minor = 0;
    cuDeviceGetName(device_name, sizeof(device_name), device);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    std::printf("device=%s compute_capability=%d.%d sm_%d%d\n", device_name, major,
                minor, major, minor);

    CUcontext context = nullptr;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuCtxCreate: %s (%d): %s\n", ErrorName(result),
                     static_cast<int>(result), ErrorText(result));
        return 5;
    }

    int load_failures = 0;
    for (int i = 1; i < argc; ++i) {
        CUmodule module = nullptr;
        result = cuModuleLoad(&module, argv[i]);
        std::printf("cubin=%s result=%s code=%d detail=%s\n", argv[i],
                    ErrorName(result), static_cast<int>(result), ErrorText(result));
        if (result == CUDA_SUCCESS) {
            cuModuleUnload(module);
        } else {
            ++load_failures;
        }
    }

    cuCtxDestroy(context);
    return load_failures == 0 ? 0 : 1;
}
