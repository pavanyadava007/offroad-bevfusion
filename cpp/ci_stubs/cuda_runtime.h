// CI-only stub of the CUDA runtime API used by trt_runner.cpp.
#pragma once
#include <cstddef>
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
enum { cudaSuccess = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaMalloc(void**, size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, int, cudaStream_t);
cudaError_t cudaMemcpy(void*, const void*, size_t, int);
cudaError_t cudaStreamCreate(cudaStream_t*);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaEventCreate(cudaEvent_t*);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
