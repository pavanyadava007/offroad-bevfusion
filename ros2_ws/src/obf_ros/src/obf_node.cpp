// ROS 2 Humble node: replays dumped nuScenes-mini frames through the TensorRT engine at a fixed rate and publishes
//   /obf/detections   vision_msgs/Detection3DArray   (CenterPoint decode of hm/reg)
//   /obf/markers      visualization_msgs/MarkerArray (boxes for rviz2)
//   /obf/drivable     nav_msgs/OccupancyGrid         (BEV-seg drivable channel; 0 = free, 100 = occupied)
//   /obf/occ_topdown  nav_msgs/OccupancyGrid         (3D occupancy collapsed over z: any non-free voxel -> occupied)
// frames are read from <replay_dir>/manifest.json as produced by `python -m obf.export.dump_samples --raw`.
#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <cmath>
#include <fstream>
#include <regex>
#include <string>
#include <vector>
#include "trt_runner.hpp"

class ObfNode : public rclcpp::Node {
 public:
  ObfNode() : Node("obf_node") {
    engine_ = declare_parameter<std::string>("engine", "results/export/bevfusion_fp16.engine");
    replay_ = declare_parameter<std::string>("replay_dir", "data/samples/replay");
    rate_ = declare_parameter<double>("rate_hz", 10.0);
    thr_ = declare_parameter<double>("score_thr", 0.3);
    frame_ = declare_parameter<std::string>("frame_id", "base_link");
    auto pr = declare_parameter<std::vector<double>>("pc_range", {-40, -40, -1, 40, 40, 5.4});
    for (int i = 0; i < 6; ++i) pc_[i] = pr[i];
    classes_ = declare_parameter<std::vector<std::string>>("classes", {"car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"});
    runner_ = std::make_unique<obf::TrtRunner>(engine_);
    load_manifest();
    det_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("/obf/detections", 10);
    mk_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/obf/markers", 10);
    drv_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/obf/drivable", 10);
    occ_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/obf/occ_topdown", 10);
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate_), [this] { step(); });
    RCLCPP_INFO(get_logger(), "engine %s, %zu frames", engine_.c_str(), frames_.size());
  }

 private:
  void load_manifest() {  // dependency-free: extract "dir": "..." entries
    std::ifstream f(replay_ + "/manifest.json");
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::regex re("\"dir\":\\s*\"([^\"]+)\"");
    for (auto it = std::sregex_iterator(s.begin(), s.end(), re); it != std::sregex_iterator(); ++it) frames_.push_back((*it)[1]);
  }
  static std::vector<float> read_f32(const std::string& p, size_t bytes) {
    std::vector<float> v(bytes / 4);
    std::ifstream f(p, std::ios::binary); f.read(reinterpret_cast<char*>(v.data()), bytes); return v;
  }
  template <class T> void grid(nav_msgs::msg::OccupancyGrid& g, int Y, int X, T fn) {
    g.header.frame_id = frame_; g.header.stamp = now();
    g.info.resolution = (pc_[3] - pc_[0]) / X; g.info.width = X; g.info.height = Y;
    g.info.origin.position.x = pc_[0]; g.info.origin.position.y = pc_[1]; g.info.origin.orientation.w = 1.0;
    g.data.resize((size_t)Y * X);
    for (int y = 0; y < Y; ++y) for (int x = 0; x < X; ++x) g.data[y * X + x] = fn(y, x);
  }
  void step() {
    if (frames_.empty()) return;
    const std::string& d = frames_[idx_++ % frames_.size()];
    std::vector<std::vector<char>> keep;
    for (const auto& t : runner_->tensors()) if (t.is_input) {
      std::ifstream f(d + "/" + t.name + ".bin", std::ios::binary);
      keep.emplace_back((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      if (keep.back().size() != t.bytes) { RCLCPP_ERROR(get_logger(), "bad input %s", t.name.c_str()); return; }
      runner_->set_input(t.name, keep.back().data());
    }
    float ms = runner_->infer();
    const auto& hm_t = runner_->tensor("hm"); const auto& reg_t = runner_->tensor("reg");
    int K = hm_t.shape[1], Y = hm_t.shape[2], X = hm_t.shape[3];
    std::vector<float> hm(hm_t.bytes / 4), reg(reg_t.bytes / 4);
    runner_->get_output("hm", hm.data()); runner_->get_output("reg", reg.data());
    auto dets = obf::decode_centerpoint(hm.data(), reg.data(), K, Y, X, pc_, thr_, 200);
    vision_msgs::msg::Detection3DArray da; da.header.frame_id = frame_; da.header.stamp = now();
    visualization_msgs::msg::MarkerArray ma;
    int id = 0;
    for (const auto& b : dets) {
      vision_msgs::msg::Detection3D det; det.header = da.header;
      det.bbox.center.position.x = b.x; det.bbox.center.position.y = b.y; det.bbox.center.position.z = b.z;
      det.bbox.center.orientation.z = std::sin(b.yaw / 2); det.bbox.center.orientation.w = std::cos(b.yaw / 2);
      det.bbox.size.x = b.l; det.bbox.size.y = b.w; det.bbox.size.z = b.h;
      vision_msgs::msg::ObjectHypothesisWithPose hyp; hyp.hypothesis.class_id = classes_[b.cls]; hyp.hypothesis.score = b.score;
      det.results.push_back(hyp); da.detections.push_back(det);
      visualization_msgs::msg::Marker m; m.header = da.header; m.ns = "obf"; m.id = id++; m.type = m.CUBE; m.action = m.ADD;
      m.pose = det.bbox.center; m.scale = det.bbox.size; m.color.a = 0.5f; m.color.r = b.cls == 8 ? 1.f : 0.f; m.color.g = b.cls == 8 ? 0.f : 1.f;
      m.lifetime = rclcpp::Duration::from_seconds(1.0 / rate_ * 1.5); ma.markers.push_back(m);
    }
    det_pub_->publish(da); mk_pub_->publish(ma);
    try {
      const auto& seg_t = runner_->tensor("seg");
      std::vector<float> seg(seg_t.bytes / 4); runner_->get_output("seg", seg.data());
      nav_msgs::msg::OccupancyGrid g; grid(g, Y, X, [&](int y, int x) { return seg[y * X + x] > 0 ? 0 : 100; }); drv_pub_->publish(g);
    } catch (...) {}
    try {
      const auto& occ_t = runner_->tensor("occ");  // [1,C,Z,Y,X]
      int C = occ_t.shape[1], Z = occ_t.shape[2];
      std::vector<float> occ(occ_t.bytes / 4); runner_->get_output("occ", occ.data());
      nav_msgs::msg::OccupancyGrid g;
      grid(g, Y, X, [&](int y, int x) {
        for (int z = 0; z < Z; ++z) {
          int best = 0; float bv = -1e9f;
          for (int c = 0; c < C; ++c) { float v = occ[(((size_t)c * Z + z) * Y + y) * X + x]; if (v > bv) { bv = v; best = c; } }
          if (best != C - 1) return (int8_t)100;  // any non-free voxel in the column
        }
        return (int8_t)0;
      });
      occ_pub_->publish(g);
    } catch (...) {}
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, "frame %zu: %zu dets, %.1f ms", idx_, dets.size(), ms);
  }

  std::string engine_, replay_, frame_; double rate_, thr_; float pc_[6]; std::vector<std::string> classes_;
  std::unique_ptr<obf::TrtRunner> runner_; std::vector<std::string> frames_; size_t idx_ = 0;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr mk_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr drv_pub_, occ_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObfNode>());
  rclcpp::shutdown();
  return 0;
}
