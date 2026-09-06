extern "C" __global__ void sm89_positive_control(unsigned int* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[0] = 0x89u;
    }
}
