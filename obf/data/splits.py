"""Scene-description based adverse-condition subsets (nuScenes scene.description contains 'rain' / 'night')."""
import json
import os
import pickle


def make_splits(infos_pkl, out_dir):
    with open(infos_pkl, "rb") as f:
        infos = pickle.load(f)
    os.makedirs(out_dir, exist_ok=True)
    subsets = {"all": [], "rain": [], "night": [], "clear_day": []}
    for s in infos["val"]:
        d = s["scene_description"].lower()
        subsets["all"].append(s["token"])
        rain, night = "rain" in d, "night" in d
        if rain:
            subsets["rain"].append(s["token"])
        if night:
            subsets["night"].append(s["token"])
        if not rain and not night:
            subsets["clear_day"].append(s["token"])
    for k, v in subsets.items():
        with open(os.path.join(out_dir, f"val_{k}.json"), "w") as f:
            json.dump(v, f)
    return {k: len(v) for k, v in subsets.items()}


if __name__ == "__main__":
    import sys
    print(make_splits(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/splits"))
