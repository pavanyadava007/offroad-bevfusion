// obf_runner <engine> <frame_dir with <input>.bin> [iters=100] [out_dir]
// Loads raw binaries dumped by `python -m obf.export.dump_samples --raw`, runs the engine, prints latency stats and
// writes outputs as <output>.bin.
#include "trt_runner.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

static std::vector<char> read_bin(const std::string& p, size_t expect) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("missing " + p);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (b.size() != expect) throw std::runtime_error(p + ": size " + std::to_string(b.size()) + " != " + std::to_string(expect));
  return b;
}

int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: obf_runner <engine> <frame_dir> [iters] [out_dir]\n"; return 1; }
  std::string engine = argv[1], dir = argv[2], out = argc > 4 ? argv[4] : "";
  int iters = argc > 3 ? std::atoi(argv[3]) : 100;
  obf::TrtRunner r(engine);
  std::vector<std::vector<char>> keep;
  for (const auto& t : r.tensors()) if (t.is_input) {
    keep.push_back(read_bin(dir + "/" + t.name + ".bin", t.bytes));
    r.set_input(t.name, keep.back().data());
  }
  for (int i = 0; i < 20; ++i) r.infer();
  std::vector<float> ms;
  for (int i = 0; i < iters; ++i) ms.push_back(r.infer());
  std::sort(ms.begin(), ms.end());
  float mean = std::accumulate(ms.begin(), ms.end(), 0.f) / ms.size();
  std::cout << "engine=" << engine << " iters=" << iters << " mean=" << mean << "ms p50=" << ms[ms.size() / 2]
            << "ms p99=" << ms[(size_t)(ms.size() * 0.99)] << "ms\n";
  if (!out.empty()) for (const auto& t : r.tensors()) if (!t.is_input) {
    std::vector<char> h(t.bytes);
    r.get_output(t.name, h.data());
    std::ofstream(out + "/" + t.name + ".bin", std::ios::binary).write(h.data(), h.size());
  }
  return 0;
}
