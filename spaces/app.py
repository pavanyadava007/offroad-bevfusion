"""offroad-bevfusion demo — cam+LiDAR+radar BEV multitask perception (ONNX Runtime CPU) + VLA action.
Runs the exported ONNX on cached nuScenes-mini frames; falls back to cached outputs when the ONNX is absent."""
import glob
import json
import os
import time

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = next((p for p in (os.path.join(_here, "assets"), os.path.join(_here, "..", "assets")) if os.path.isdir(p)),
            os.path.join(_here, "assets"))
ONNX = os.path.join(ROOT, "bevfusion.onnx")
FRAMES = sorted(glob.glob(os.path.join(ROOT, "samples", "*")))
PC = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
VEHICLES = {0, 1, 2, 3, 4}

# validated categorical slots on the dark surface (dataviz reference palette, all-pairs safe)
C_DRIV, C_VEH, C_PED = (0x39, 0x87, 0xE5), (0x19, 0x9E, 0x70), (0xD9, 0x59, 0x26)
C_OTHER = (0x8A, 0x89, 0x80)
SURF, GRID_C, INK = (0x14, 0x14, 0x13), (0x30, 0x30, 0x2E), (0xC3, 0xC2, 0xB7)
STATUS = {"stop": ("#e66767", "&#9632; STOP"), "wait_for_person": ("#c98500", "&#9888; WAIT FOR PERSON"),
          "reverse": ("#c98500", "&#9666; REVERSE"), "approach_pile": ("#199e70", "&#9654; APPROACH PILE"),
          "dump": ("#199e70", "&#9660; DUMP")}
_sess = None


def sess():
    global _sess
    if _sess is None and os.path.exists(ONNX):
        import onnxruntime as ort
        _sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    return _sess


def run(frame_dir):
    s = sess()
    if s is not None:
        feed = {i.name: np.load(os.path.join(frame_dir, i.name + ".npy")) for i in s.get_inputs()}
        t0 = time.perf_counter()
        outs = dict(zip([o.name for o in s.get_outputs()], s.run(None, feed)))
        return outs, time.perf_counter() - t0
    outs = {}
    for k in ("hm", "reg", "seg", "occ"):
        p = os.path.join(frame_dir, k + ".npy")
        if os.path.exists(p):
            outs[k] = np.load(p)
    return outs, None


def decode_dets(o, thr=0.35, topk=12):
    if "hm" not in o:
        return []
    hm = 1 / (1 + np.exp(-o["hm"][0])); reg = o["reg"][0]
    K, Y, X = hm.shape
    # 3x3 max-pool NMS
    pad = np.pad(hm, ((0, 0), (1, 1), (1, 1)), constant_values=-1)
    peak = np.ones_like(hm, bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            peak &= hm >= pad[:, dy:dy + Y, dx:dx + X]
    dets = []
    for c, y, x in zip(*np.where((hm > thr) & peak)):
        r = reg[:, y, x]
        dets.append({"cls": int(c), "score": float(hm[c, y, x]),
                     "x": float((x + r[0]) * 0.4 + PC[0]), "y": float((y + r[1]) * 0.4 + PC[1]),
                     "w": float(np.exp(min(r[3], 6))), "l": float(np.exp(min(r[4], 6))),
                     "yaw": float(np.arctan2(r[6], r[7]))})
    return sorted(dets, key=lambda d: -d["score"])[:topk]


def _disp(grid):  # ego grid [iy, ix] -> display array [r, c]: forward(+x) up, left(+y) left
    return grid[::-1, ::-1].T[::-1, :].copy() if False else np.flipud(np.fliplr(grid.T))


def _rings(draw, S, scale, color=GRID_C):
    cx = cy = S // 2
    for m in (10, 20, 30, 40):
        r = m * scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=1)
        draw.text((cx + 4, cy - r + 3), f"{m} m", fill=(0x7A, 0x79, 0x71))
    draw.line([cx, 0, cx, S], fill=color, width=1)
    draw.line([0, cy, S, cy], fill=color, width=1)


def _ego(draw, S):
    cx = cy = S // 2
    draw.polygon([(cx, cy - 10), (cx - 6, cy + 8), (cx + 6, cy + 8)], fill=(0xFF, 0xFF, 0xFF))


def render_bev(o, dets, gt_peds=None, S=640):
    canvas = np.zeros((200, 200, 3), np.float32) + np.array(SURF, np.float32)
    if "seg" in o:
        seg = 1 / (1 + np.exp(-o["seg"][0]))  # [3,Y,X] drivable/vehicle/pedestrian
        for ch, col, a in ((0, C_DRIV, 0.40), (1, C_VEH, 0.85), (2, C_PED, 0.95)):
            m = _disp(seg[ch] > 0.5).astype(np.float32) * a
            canvas = canvas * (1 - m[..., None]) + np.array(col, np.float32) * m[..., None]
    img = Image.fromarray(canvas.astype(np.uint8)).resize((S, S), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    scale = S / 80.0  # px per metre
    _rings(draw, S, scale)
    for d in dets:
        col = C_PED if d["cls"] == 8 else (C_VEH if d["cls"] in VEHICLES else C_OTHER)
        cx, cy = S / 2 - d["y"] * scale, S / 2 - d["x"] * scale  # left=+y, up=+x
        c, s = np.cos(d["yaw"]), np.sin(d["yaw"])
        pts = []
        for dx, dy in ((d["l"] / 2, d["w"] / 2), (d["l"] / 2, -d["w"] / 2), (-d["l"] / 2, -d["w"] / 2), (-d["l"] / 2, d["w"] / 2)):
            ex, ey = d["x"] + dx * c - dy * s, d["y"] + dx * s + dy * c
            pts.append((S / 2 - ey * scale, S / 2 - ex * scale))
        draw.polygon(pts, outline=tuple(col), width=2)
    for p in (gt_peds or []):
        cx, cy = S / 2 - p["y"] * (S / 80.0), S / 2 - p["x"] * (S / 80.0)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=(255, 255, 255), width=2)
    hz = [p for p in (gt_peds or []) if p["in_path"] and p["dist"] < 10]
    if hz:
        p = min(hz, key=lambda q: q["dist"])
        cx, cy = S / 2 - p["y"] * (S / 80.0), S / 2 - p["x"] * (S / 80.0)
        draw.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], outline=(0xE6, 0x67, 0x67), width=3)
        draw.text((cx + 20, cy - 8), f"in path · {p['dist']} m", fill=(0xE6, 0x67, 0x67))
    _ego(draw, S)
    return img


def render_occ(o, S=640):
    img = Image.new("RGB", (S, S), SURF)
    if "occ" in o:
        cls = o["occ"][0].argmax(0)          # [Z,Y,X] class ids, 17 = free
        occ = cls != 17                      # [Z,Y,X]
        Z = occ.shape[0]
        hts = np.where(occ.any(0), occ.shape[0] - 1 - np.argmax(occ[::-1], 0), -1)  # top occupied voxel per column
        h = _disp(hts.astype(np.float32))
        base = np.zeros((200, 200, 3), np.float32) + np.array(SURF, np.float32)
        m = h >= 0
        t = np.clip(h / (Z - 1), 0, 1)[..., None]                                   # sequential single-hue ramp (height)
        lo, hi = np.array((0x17, 0x3A, 0x63), np.float32), np.array((0x8F, 0xC2, 0xFF), np.float32)
        base[m] = lo + (hi - lo) * t[m]
        img = Image.fromarray(base.astype(np.uint8)).resize((S, S), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    _rings(draw, S, S / 80.0)
    _ego(draw, S)
    return img


def frame_label(d):
    v = json.load(open(os.path.join(d, "vla.json"))) if os.path.exists(os.path.join(d, "vla.json")) else {}
    ped = v.get("nearest_pedestrian_in_path_m")
    n = len(v.get("gt_pedestrians", []))
    tag = f"ped IN PATH {ped} m" if ped is not None else f"{n} peds, none in corridor"
    return f"{os.path.basename(d)[:2]} · {tag} · {v.get('action', '?').upper()}"


LABELS = {frame_label(d): d for d in FRAMES}


def render(label):
    d = LABELS[label]
    o, dt = run(d)
    dets = decode_dets(o)
    vla_pre = json.load(open(os.path.join(d, "vla.json"))) if os.path.exists(os.path.join(d, "vla.json")) else {}
    bev = render_bev(o, dets, vla_pre.get("gt_pedestrians"))
    occ = render_occ(o)
    cam = os.path.join(d, "cam_front.jpg")
    rows = [[CLASSES[x["cls"]], round(x["score"], 2), round(float(np.hypot(x["x"], x["y"])), 1),
             round(float(np.degrees(np.arctan2(x["y"], x["x"]))), 0), round(x["x"], 1), round(x["y"], 1)] for x in dets]
    vla = json.load(open(os.path.join(d, "vla.json"))) if os.path.exists(os.path.join(d, "vla.json")) else {}
    color, badge = STATUS.get(vla.get("action", ""), ("#8a8980", vla.get("action", "n/a")))
    ped = vla.get("nearest_pedestrian_in_path_m")
    vla_html = f"""
    <div class="card vla">
      <div class="cap">VLA task grounding — Qwen2.5-VL-3B + LoRA (cached, GT perception)</div>
      <div class="badge" style="background:{color}1f;border:1px solid {color};color:{color}">{badge}</div>
      <div class="reason">{vla.get('reason', '')}</div>
    </div>"""
    driv = float((1 / (1 + np.exp(-o["seg"][0][0])) > 0.5).mean()) if "seg" in o else 0.0
    tiles = f"""
    <div class="tiles">
      <div class="tile"><div class="v">{(f"{dt*1000:.0f} ms" if dt else "cached")}</div><div class="k">CPU inference (this run)</div></div>
      <div class="tile"><div class="v">{len(dets)}</div><div class="k">detections &gt; 0.35</div></div>
      <div class="tile"><div class="v">{driv*100:.0f}%</div><div class="k">BEV drivable coverage</div></div>
      <div class="tile"><div class="v">{(f"{ped} m" if ped is not None else "—")}</div><div class="k">nearest pedestrian in path</div></div>
    </div>"""
    return cam, bev, occ, rows, tiles, vla_html, vla


CSS = """
.gradio-container {max-width: 1280px !important; margin: 0 auto;}
#hdr h1 {font-size: 1.45rem; margin: 0 0 2px;}
#hdr .sub {color: #a8a79e; font-size: .92rem;}
#hdr .chips {margin-top: 8px;}
#hdr .chip, .legend .chip {display: inline-block; border: 1px solid #3a3a38; border-radius: 999px;
  padding: 2px 10px; margin-right: 6px; font-size: .78rem; color: #c3c2b7;}
.legend .dot {display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; vertical-align: -1px;}
.tiles {display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 4px 0 2px;}
.tile {border: 1px solid #2e2e2c; border-radius: 10px; padding: 10px 14px; background: #1a1a19;}
.tile .v {font-size: 1.35rem; font-weight: 650; color: #fff;}
.tile .k {font-size: .78rem; color: #a8a79e; margin-top: 2px;}
.card.vla {border: 1px solid #2e2e2c; border-radius: 10px; padding: 12px 14px; background: #1a1a19;}
.card.vla .cap {font-size: .78rem; color: #a8a79e; margin-bottom: 8px;}
.card.vla .badge {display: inline-block; font-weight: 700; letter-spacing: .04em; border-radius: 8px; padding: 6px 14px; font-size: 1.05rem;}
.card.vla .reason {margin-top: 8px; color: #c3c2b7; font-size: .9rem;}
footer {display: none !important;}
"""

with gr.Blocks(title="offroad-bevfusion") as demo:
    gr.HTML("""
    <div id="hdr">
      <h1>offroad-bevfusion</h1>
      <div class="sub">Camera + LiDAR + radar &rarr; one 200&times;200 BEV @ 0.4 m &middot; 3D detection &middot; BEV segmentation &middot; occupancy &middot; VLA task grounding</div>
      <div class="chips">
        <span class="chip">ONNX opset 17 &middot; static shapes</span><span class="chip">ONNX Runtime CPU &middot; ~0.85 s/frame measured</span>
        <span class="chip">TensorRT FP16 on L4: 15.3 ms</span><span class="chip"><a href="https://github.com/pavanyadava007/offroad-bevfusion" style="color:#3987e5;text-decoration:none">GitHub repo &nearr;</a></span>
      </div>
    </div>""")
    with gr.Tab("Frame explorer"):
        sel = gr.Radio(list(LABELS), value=list(LABELS)[0], label="nuScenes-mini frame (cached)")
        tiles = gr.HTML()
        with gr.Row():
            with gr.Column(scale=5):
                cam = gr.Image(label="CAM_FRONT — GT pedestrian boxes (red = in corridor < 10 m, green = elsewhere)", height=300)
                vla = gr.HTML()
            with gr.Column(scale=4):
                bev = gr.Image(label="BEV — segmentation + detections (forward ↑)", height=430)
                gr.HTML("""<div class="legend">
                  <span class="chip"><span class="dot" style="background:#3987e5"></span>drivable</span>
                  <span class="chip"><span class="dot" style="background:#199e70"></span>vehicle</span>
                  <span class="chip"><span class="dot" style="background:#d95926"></span>pedestrian</span>
                  <span class="chip"><span class="dot" style="background:#8a8980"></span>other det</span>
                  <span class="chip">○ GT pedestrian</span>
                  <span class="chip">△ ego</span></div>""")
            with gr.Column(scale=4):
                occ = gr.Image(label="3D occupancy — top surface height (forward ↑)", height=430)
                gr.HTML('<div class="legend"><span class="chip"><span class="dot" style="background:#173a63"></span>low&nbsp;&rarr;&nbsp;<span class="dot" style="background:#8fc2ff"></span>high occupied</span></div>')
        dets = gr.Dataframe(headers=["class", "conf", "dist m", "bearing °", "x m", "y m"], label="detections (score > 0.35, top 12)",
                            interactive=False)
        raw = gr.JSON(label="raw VLA JSON", open=False)
    with gr.Tab("Sequence replay (81 val frames)"):
        gr.HTML('<div class="sub" style="color:#a8a79e;font-size:.9rem;margin:6px 0">The full nuScenes-mini validation split replayed '
                'through the model (TensorRT-equivalent PyTorch forward): front camera + BEV segmentation/detections, forward-up. '
                'Pre-rendered offline; the ROS 2 node runs the same stream live at 11.5 ms/frame on an L4 (see docs/rviz_replay.gif in the repo).</div>')
        gr.Image(value=os.path.join(ROOT, "replay.gif"), label="validation sequence replay", height=420)
    gr.HTML('<div class="sub" style="color:#8a8980;font-size:.8rem;margin-top:6px">Measured CPU inference ~0.85 s/frame mean '
            '(0.81–0.96 s, 32-core x86; HF free-tier CPUs will be slower). Tensors regenerable via scripts/make_demo_assets.py. '
            'VLA JSONs are cached offline outputs of Qwen2.5-VL-3B + LoRA with GT perception.</div>')
    outs = [cam, bev, occ, dets, tiles, vla, raw]
    sel.change(render, sel, outs)
    demo.load(render, sel, outs)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Base(primary_hue="blue", neutral_hue="stone"))
