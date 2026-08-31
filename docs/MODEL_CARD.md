# Model card — offroad-bevfusion (cam+LiDAR+radar, nuScenes-mini)

| | |
|---|---|
| Inputs | 6×RGB 256×704; LiDAR pillars [20000×20×5]; radar pillars [1500×12×7]; precomputed cam→BEV indices |
| BEV | 200×200 @ 0.4 m, ego frame (x fwd, y left) |
| Heads | CenterPoint (10 cls, 10 reg ch), BEV seg (3 multilabel), occupancy 18×16×200×200 |
| Params | ResNet-18 camera backbone (11.7 M) + BEV net (~9 M) + heads (~5 M) |
| Training | AdamW 2e-4, cosine, AMP fp16, 12 epochs on 323 frames (local NVIDIA L4), Kendall task weighting |
| Export | ONNX opset 17 static; TRT FP16; TRT INT8 (IInt8EntropyCalibrator2, 100 frames) |
| Known limits | mini split → high variance; 40 m range; max-pool NMS (no rotated NMS); Orin not measured |
| Intended use | portfolio / research; **not** for on-vehicle safety functions |

Numbers: see `docs/REPORT.md` (generated; empty until the GPU runs are done).
