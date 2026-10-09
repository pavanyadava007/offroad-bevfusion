#pragma once
// Synthetic GPU co-tenant load: many short FMA-bound blocks so the block scheduler can interleave a higher-priority
// stream at block granularity. Used by gpu_hog (separate process) and latency_bench --inproc-hog (same process).
#include <cuda_runtime.h>

namespace obf::rt {

// Launches one hog kernel of `blocks` x 256 threads, each thread doing `inner` dependent FMA iterations on buf.
// buf must hold at least blocks * 256 floats. Asynchronous on `stream`.
cudaError_t launch_hog(cudaStream_t stream, float* buf, int blocks, int inner);

}  // namespace obf::rt
