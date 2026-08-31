"""Qwen2.5-VL-3B-Instruct -> JSON action primitive. Fail-safe: unparsable / invalid output -> 'stop'."""
import json
import re

import torch

from .prompt import ACTIONS, SYSTEM, user_prompt


class VLAGrounder:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", lora=None, device="cuda", dtype=torch.bfloat16, max_new_tokens=96):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def messages(self, image, perception, task):
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": user_prompt(json.dumps(perception, separators=(",", ":")), task)}]}]

    @torch.no_grad()
    def __call__(self, image, perception, task):
        from qwen_vl_utils import process_vision_info
        msgs = self.messages(image, perception, task)
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = self.processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        raw = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return parse_action(raw)


def parse_action(raw):
    m = re.search(r"\{.*?\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        if d.get("action") in ACTIONS:
            return {"action": d["action"], "target": d.get("target"), "reason": d.get("reason", ""), "raw": raw, "parsed": True}
    except json.JSONDecodeError:
        pass
    return {"action": "stop", "target": None, "reason": "fail-safe: unparsable VLM output", "raw": raw, "parsed": False}
