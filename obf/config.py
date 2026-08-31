"""Minimal YAML config with `_base_` inheritance and `key.sub=value` CLI overrides."""
import os

import yaml


class Cfg(dict):
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    __setattr__ = dict.__setitem__

    def get_path(self, dotted, default=None):
        d = self
        for k in dotted.split("."):
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = _merge(a[k], v)
        else:
            a[k] = v
    return a


def load_cfg(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "_base_" in cfg:
        base = load_cfg(os.path.join(os.path.dirname(path), cfg.pop("_base_")))
        cfg = _merge(dict(base), cfg)
    for o in overrides or []:
        k, v = o.split("=", 1)
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = yaml.safe_load(v)
    return Cfg(cfg)
