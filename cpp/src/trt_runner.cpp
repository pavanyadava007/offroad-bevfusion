#include "trt_runner.hpp"
#include <NvInferPlugin.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e)); } while (0)

namespace obf {

void TrtRunner::Logger::log(Severity s, const char* msg) noexcept {
  if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
}

size_t dtype_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kBOOL: return 1;
    case nvinfer1::DataType::kINT8: return 1;
    default: return 4;
  }
}

TrtRunner::TrtRunner(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open engine " + path);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  initLibNvInferPlugins(&logger_, "");  // engines use TRT's bundled ScatterReduction plugin
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) throw std::runtime_error("engine deserialization failed");
  ctx_.reset(engine_->createExecutionContext());
  CK(cudaStreamCreate(&stream_));
  CK(cudaEventCreate(&t0_)); CK(cudaEventCreate(&t1_));
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    TensorInfo t;
    t.name = engine_->getIOTensorName(i);
    auto dims = engine_->getTensorShape(t.name.c_str());
    size_t n = 1;
    for (int d = 0; d < dims.nbDims; ++d) { t.shape.push_back(dims.d[d]); n *= dims.d[d]; }
    t.dtype = engine_->getTensorDataType(t.name.c_str());
    t.bytes = n * dtype_size(t.dtype);
    t.is_input = engine_->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    CK(cudaMalloc(&t.dev, t.bytes));
    ctx_->setTensorAddress(t.name.c_str(), t.dev);
    tensors_.push_back(t);
  }
}

TrtRunner::~TrtRunner() {
  for (auto& t : tensors_) cudaFree(t.dev);
  cudaEventDestroy(t0_); cudaEventDestroy(t1_); cudaStreamDestroy(stream_);
}

const TensorInfo& TrtRunner::tensor(const std::string& name) const {
  for (auto& t : tensors_) if (t.name == name) return t;
  throw std::runtime_error("no tensor " + name);
}

void TrtRunner::set_input(const std::string& name, const void* host) {
  const auto& t = tensor(name);
  CK(cudaMemcpyAsync(t.dev, host, t.bytes, cudaMemcpyHostToDevice, stream_));
}

void TrtRunner::get_output(const std::string& name, void* host) const {
  const auto& t = tensor(name);
  CK(cudaMemcpy(host, t.dev, t.bytes, cudaMemcpyDeviceToHost));
}

float TrtRunner::infer() {
  CK(cudaEventRecord(t0_, stream_));
  if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
  CK(cudaEventRecord(t1_, stream_));
  CK(cudaStreamSynchronize(stream_));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, t0_, t1_));
  return ms;
}

std::vector<Det> decode_centerpoint(const float* hm, const float* reg, int K, int Y, int X, const float pc_range[6],
                                    float score_thr, int max_dets) {
  const int YX = Y * X;
  const float dx = (pc_range[3] - pc_range[0]) / X, dy = (pc_range[4] - pc_range[1]) / Y;
  auto sig = [](float v) { return 1.f / (1.f + std::exp(-v)); };
  std::vector<Det> dets;
  for (int k = 0; k < K; ++k) {
    const float* h = hm + (size_t)k * YX;
    for (int y = 0; y < Y; ++y) for (int x = 0; x < X; ++x) {
      float v = h[y * X + x];
      float s = sig(v);
      if (s < score_thr) continue;
      bool peak = true;  // 3x3 max-pool NMS
      for (int ddy = -1; ddy <= 1 && peak; ++ddy) for (int ddx = -1; ddx <= 1; ++ddx) {
        int yy = y + ddy, xx = x + ddx;
        if (yy < 0 || yy >= Y || xx < 0 || xx >= X) continue;
        if (h[yy * X + xx] > v) { peak = false; break; }
      }
      if (!peak) continue;
      auto R = [&](int c) { return reg[(size_t)c * YX + y * X + x]; };
      Det d;
      d.x = (x + R(0)) * dx + pc_range[0]; d.y = (y + R(1)) * dy + pc_range[1]; d.z = R(2);
      d.w = std::exp(std::min(R(3), 6.f)); d.l = std::exp(std::min(R(4), 6.f)); d.h = std::exp(std::min(R(5), 6.f));
      d.yaw = std::atan2(R(6), R(7)); d.vx = R(8); d.vy = R(9); d.score = s; d.cls = k;
      dets.push_back(d);
    }
  }
  std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b) { return a.score > b.score; });
  if ((int)dets.size() > max_dets) dets.resize(max_dets);
  return dets;
}

}  // namespace obf
