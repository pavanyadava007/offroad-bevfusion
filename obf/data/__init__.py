from .nuscenes_mini import NuScenesDataset, collate  # noqa: F401


def build_dataset(cfg, split, tokens=None):
    if cfg.get("dataset") == "fake":
        from .fake import FakeDataset
        return FakeDataset(cfg, split, n=cfg.data.get("fake_n", 8))
    if cfg.get("dataset", "nuscenes") == "goose":
        from .goose import GooseDataset
        return GooseDataset(cfg, split)
    return NuScenesDataset(cfg, split, tokens)
