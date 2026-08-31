#pragma once
// Minimal TensorRT-10 C++ inference wrapper (enqueueV3, cudaEvent timing). Used by obf_runner and the ROS 2 node.
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace obf {

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  nvinfer1::DataType dtype;
  size_t bytes = 0;
  bool is_input = false;
  void* dev = nullptr;
};

size_t dtype_size(nvinfer1::DataType t);

class TrtRunner {
 public:
  explicit TrtRunner(const std::string& engine_path);
  ~TrtRunner();
  TrtRunner(const TrtRunner&) = delete;
  TrtRunner& operator=(const TrtRunner&) = delete;

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo& tensor(const std::string& name) const;
  void set_input(const std::string& name, const void* host);  // H2D copy
  void get_output(const std::string& name, void* host) const;  // D2H copy
  float infer();  // returns GPU time in ms

 private:
  class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override;
  } logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
  std::vector<TensorInfo> tensors_;
  cudaStream_t stream_{};
  cudaEvent_t t0_{}, t1_{};
};

// CenterPoint-style decode on host: hm [K,Y,X] (logits), reg [10,Y,X] -> boxes.
struct Det { float x, y, z, w, l, h, yaw, vx, vy, score; int cls; };
std::vector<Det> decode_centerpoint(const float* hm, const float* reg, int K, int Y, int X, const float pc_range[6],
                                    float score_thr, int max_dets);

}  // namespace obf
