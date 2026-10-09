#include "hog_kernel.cuh"

namespace obf::rt {

__global__ void hog_kernel(float* buf, int inner) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float a = buf[i], b = 1.0001f, c = 0.9999f;
  for (int k = 0; k < inner; ++k) {
    a = fmaf(a, b, c);
    b = fmaf(b, c, a * 1e-7f);
  }
  buf[i] = a + b;  // keep the result live so the loop is not optimised away
}

cudaError_t launch_hog(cudaStream_t stream, float* buf, int blocks, int inner) {
  hog_kernel<<<blocks, 256, 0, stream>>>(buf, inner);
  return cudaGetLastError();
}

}  // namespace obf::rt
