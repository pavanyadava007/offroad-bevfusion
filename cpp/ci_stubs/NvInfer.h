// CI-only stub of the TensorRT 10 API surface used by trt_runner.cpp (syntax check on runners without CUDA/TensorRT).
#pragma once
#include <cstddef>
#include <cstdint>
namespace nvinfer1 {
enum class DataType { kFLOAT, kHALF, kINT8, kINT32, kBOOL, kUINT8, kFP8, kBF16, kINT64, kINT4 };
enum class TensorIOMode { kNONE, kINPUT, kOUTPUT };
struct Dims { int32_t nbDims; int64_t d[8]; };
class ILogger {
 public:
  enum class Severity { kINTERNAL_ERROR, kERROR, kWARNING, kINFO, kVERBOSE };
  virtual void log(Severity, const char*) noexcept = 0;
  virtual ~ILogger() {}
};
class IExecutionContext {
 public:
  bool setTensorAddress(const char*, void*);
  bool enqueueV3(void*);
};
class ICudaEngine {
 public:
  int32_t getNbIOTensors() const;
  const char* getIOTensorName(int32_t) const;
  Dims getTensorShape(const char*) const;
  DataType getTensorDataType(const char*) const;
  TensorIOMode getTensorIOMode(const char*) const;
  IExecutionContext* createExecutionContext();
};
class IRuntime {
 public:
  ICudaEngine* deserializeCudaEngine(const void*, size_t);
};
IRuntime* createInferRuntime(ILogger&);
}  // namespace nvinfer1
