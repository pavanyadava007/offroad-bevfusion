"""HF Spaces (free CPU) demo: ONNX Runtime on cached nuScenes-mini frames -> BEV seg / detections / occupancy top view,
plus the cached VLA JSON. Assets: assets/samples/<k>/<input>.npy (+ cam_front.jpg, vla.json), assets/bevfusion.onnx (git-lfs)."""
import glob
import json
import os
import numpy as np
import gradio as gr

ROOT = os.path.join(os.path.dirname(__file__), "..", "assets")
ONNX = os.path.join(ROOT, "bevfusion.onnx")
FRAMES = sorted(glob.glob(os.path.join(ROOT, "samples", "*")))
PC = (-40, -40, -1, 40, 40, 5.4)
_sess = None


def sess():
    global _sess
    if _sess is None and os.path.exists(ONNX):
        import onnxruntime as ort
        _sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    return _sess


def run(frame_dir):
    outs = {}
    s = sess()
    if s is not None:
        feed = {i.name: np.load(os.path.join(frame_dir, i.name + ".npy")) for i in s.get_inputs()}
        outs = dict(zip([o.name for o in s.get_outputs()], s.run(None, feed)))
    else:  # cached outputs
        for k in ("hm", "reg", "seg", "occ"):
            p = os.path.join(frame_dir, k + ".npy")
            if os.path.exists(p):
                outs[k] = np.load(p)
    return outs


def render(frame_dir):
    o = run(frame_dir)
    seg = 1 / (1 + np.exp(-o["seg"][0])) if "seg" in o else np.zeros((3, 200, 200))
    rgb = np.stack([seg[1], seg[0], seg[2]], -1)  # vehicle=R, drivable=G, pedestrian=B
    img = (np.flipud(rgb) * 255).astype(np.uint8)
    dets = []
    if "hm" in o:
        hm = 1 / (1 + np.exp(-o["hm"][0])); reg = o["reg"][0]
        for c in range(hm.shape[0]):
            ys, xs = np.where(hm[c] > 0.3)
            for y, x in zip(ys, xs):
                dets.append({"cls": int(c), "score": round(float(hm[c, y, x]), 2), "x": round(float((x + reg[0, y, x]) * 0.4 - 40), 1),
                             "y": round(float((y + reg[1, y, x]) * 0.4 - 40), 1)})
    occ_top = None
    if "occ" in o:
        cls = o["occ"][0].argmax(0)  # [Z,Y,X]
        occ_top = np.flipud((cls != 17).any(0).astype(np.uint8) * 255)
    vla = os.path.join(frame_dir, "vla.json")
    vla = json.load(open(vla)) if os.path.exists(vla) else {"note": "no cached VLA output for this frame"}
    cam = os.path.join(frame_dir, "cam_front.jpg")
    return (cam if os.path.exists(cam) else None), img, occ_top, sorted(dets, key=lambda d: -d["score"])[:20], vla


with gr.Blocks(title="offroad-bevfusion") as demo:
    gr.Markdown("## offroad-bevfusion — cam+LiDAR+radar BEV perception (ONNX Runtime, CPU) + VLA action")
    gr.Markdown("_Measured CPU inference: ~0.85 s/frame mean (0.81–0.96 s over the 5 cached frames, ONNX Runtime CPU EP, 32-core x86; HF free-tier CPUs will be slower). VLA JSONs are cached offline outputs of Qwen2.5-VL-3B + LoRA with GT perception._")
    sel = gr.Dropdown(FRAMES, value=FRAMES[0] if FRAMES else None, label="cached nuScenes-mini frame")
    with gr.Row():
        cam = gr.Image(label="CAM_FRONT"); seg = gr.Image(label="BEV seg (G drivable, R vehicle, B pedestrian)"); occ = gr.Image(label="occupancy top view")
    dets = gr.JSON(label="detections (ego frame)"); vla = gr.JSON(label="VLA action JSON")
    sel.change(render, sel, [cam, seg, occ, dets, vla])
    if FRAMES:
        demo.load(render, sel, [cam, seg, occ, dets, vla])

if __name__ == "__main__":
    demo.launch()
