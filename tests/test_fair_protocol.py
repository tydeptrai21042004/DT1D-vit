import json
from pathlib import Path
from types import SimpleNamespace

import torch


def test_vtab_training_never_uses_trainval_for_checkpoint_selection(monkeypatch):
    import train

    calls = []
    dummy = object()
    monkeypatch.setattr(train.data_loader, "construct_train_loader", lambda cfg: calls.append("train") or dummy)
    monkeypatch.setattr(train.data_loader, "construct_trainval_loader", lambda cfg: calls.append("trainval") or dummy)
    monkeypatch.setattr(train.data_loader, "construct_val_loader", lambda cfg: calls.append("val") or dummy)
    monkeypatch.setattr(train.data_loader, "construct_test_loader", lambda cfg: calls.append("test") or dummy)

    cfg = SimpleNamespace(DATA=SimpleNamespace(NAME="vtab-caltech101", NO_TEST=False))
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    loaders = train.get_loaders(cfg, logger)
    assert loaders == (dummy, dummy, dummy)
    assert calls == ["train", "val", "test"]
    assert "trainval" not in calls


def test_pfeiffer_initialization_is_near_identity_but_not_dead():
    from src.configs.config import get_cfg
    from src.configs import vit_configs
    from src.models.vit_adapter.vit import ADPT_Block

    torch.manual_seed(7)
    cfg = get_cfg()
    cfg.MODEL.ADAPTER.NAME = "Pfeiffer"
    cfg.MODEL.ADAPTER.REDUCTION_FACTOR = 16
    block = ADPT_Block(vit_configs.get_b16_config(), False, cfg.MODEL.ADAPTER, grid_size=(14, 14))
    assert torch.count_nonzero(block.adapter_downsample.weight).item() > 0
    assert torch.count_nonzero(block.adapter_upsample.weight).item() == 0
    assert torch.count_nonzero(block.adapter_downsample.bias).item() == 0
    assert torch.count_nonzero(block.adapter_upsample.bias).item() == 0


def test_same_seed_gives_same_classifier_head_even_after_different_rng_consumption():
    from src.configs.config import get_cfg
    from src.models.vit_models import ViT

    class Dummy:
        feat_dim = 16

    cfg = get_cfg()
    cfg.SEED = 2
    cfg.DATA.NUMBER_CLASSES = 5
    cfg.MODEL.MLP_NUM = 0

    a = Dummy()
    b = Dummy()
    torch.manual_seed(11)
    _ = torch.randn(3)  # method A consumed a little RNG
    ViT.setup_head(a, cfg)
    torch.manual_seed(11)
    _ = torch.randn(1000)  # method B consumed much more RNG
    ViT.setup_head(b, cfg)
    assert torch.equal(a.head.last_layer.weight, b.head.last_layer.weight)
    assert torch.equal(a.head.last_layer.bias, b.head.last_layer.bias)


def test_dataloader_rng_is_method_independent(monkeypatch):
    from src.configs.config import get_cfg
    from src.data import loader

    class DummyDataset(torch.utils.data.Dataset):
        name = "dummy"
        def __init__(self, cfg, split):
            self.split = split
        def __len__(self):
            return 12
        def __getitem__(self, idx):
            # Emulates a stochastic transform; values must match across methods.
            return {"image": torch.rand(1), "label": idx}

    monkeypatch.setitem(loader._DATASET_CATALOG, "dummy", DummyDataset)
    cfg = get_cfg()
    cfg.DATA.NAME = "dummy"
    cfg.DATA.BATCH_SIZE = 4
    cfg.DATA.NUM_WORKERS = 0
    cfg.SEED = 1

    torch.manual_seed(123)
    _ = torch.randn(200)  # emulate one method consuming RNG before loader iteration
    l1 = loader.construct_train_loader(cfg)
    out1 = [(b["label"].clone(), b["image"].clone()) for b in l1]

    torch.manual_seed(999)
    _ = torch.randn(7)  # emulate a different architecture
    l2 = loader.construct_train_loader(cfg)
    out2 = [(b["label"].clone(), b["image"].clone()) for b in l2]

    assert len(out1) == len(out2)
    for (y1, x1), (y2, x2) in zip(out1, out2):
        assert torch.equal(y1, y2)
        assert torch.equal(x1, x2)


def test_fair_config_generator_gives_every_method_same_search_budget(tmp_path):
    import run_fair_vit_comparison as fair

    kw = dict(
        dataset="flowers102",
        data_path=str(tmp_path / "data"),
        model_root=str(tmp_path / "weights"),
        output_root=str(tmp_path / "out"),
        resolution=224,
        num_workers=0,
        weight_decay=1e-4,
        warmup_epoch=1,
        epochs=10,
        patience=20,
        log_every=10,
        vpt_tokens=5,
        pfeiffer_reduction=16,
        dt1d_active_offsets=fair.active_offsets_for_repo_schema(),
        lr_grid=None,
    )
    for m in fair.METHOD_ORDER:
        kw[f"{m}_lr_grid"] = None
    args = SimpleNamespace(**kw)

    methods = fair.METHOD_ORDER
    grids = {(32, m): fair.resolve_method_lr_grid(args, m, 32) for m in methods}
    assert {len(g) for g in grids.values()} == {10}

    configs = {}
    for m in methods:
        for lr in grids[(32, m)]:
            configs[(32, m, lr)] = fair.make_config(args, 32, m, lr, "tune")
    audit = fair.audit_tuning_configs(configs, methods, grids)
    assert audit["status"] == "PASS"
    assert audit["batches"]["32"]["trials_per_method"] == 10

    assert fair.method_optimizer("vpt") == "sgd"
    assert fair.method_optimizer("linear") == "sgd"
    assert fair.method_optimizer("full") == "adamw"
    expected_vpt = sorted({x * 32 / 256.0 for x in fair.VPT_NOMINAL_LRS})
    assert grids[(32, "vpt")] == expected_vpt

    for m in methods:
        for lr in grids[(32, m)]:
            cfg = configs[(32, m, lr)]
            assert cfg["DATA"]["NO_TEST"] is True
            assert cfg["SOLVER"]["OPTIMIZER"] == fair.method_optimizer(m)
            assert cfg["SOLVER"]["WEIGHT_DECAY"] == 1e-4
            assert cfg["SOLVER"]["TOTAL_EPOCH"] == 10
            assert cfg["SOLVER"]["BASE_LR"] == lr


def test_tuning_selection_uses_validation_only():
    import run_fair_vit_comparison as fair

    records = [
        (1e-4, {"best_val_top1": 0.60, "test_top1": 0.99}),
        (1e-3, {"best_val_top1": 0.80, "test_top1": 0.10}),
        (5e-3, {"best_val_top1": 0.70, "test_top1": 1.00}),
    ]
    lr, val = fair.select_lr(records)
    assert lr == 1e-3
    assert val == 0.80


def test_trainer_restores_best_validation_state_and_tests_once():
    from src.configs.config import get_cfg
    from src.engine.trainer import Trainer

    class TinyDataset(torch.utils.data.Dataset):
        name = "tiny"
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return {"image": torch.tensor([[0.0]]), "label": torch.tensor(0)}
        def get_class_weights(self, _):
            return [1.0, 1.0]

    class TinyEvaluator:
        def __init__(self):
            self.results = {}
            self.epoch = 0
        def update_iteration(self, epoch):
            self.epoch = int(epoch)

    class TinyTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.val_values = [0.2, 0.8, 0.4]
            self.test_calls = 0
            self.test_weight_seen = None
        def save_prompt(self, epoch):
            pass
        def forward_one_batch(self, inputs, targets, is_train):
            if is_train:
                with torch.no_grad():
                    self.model.weight.add_(1.0)
            return torch.tensor(1.0), torch.zeros((1, 2))
        def eval_classifier(self, data_loader, prefix, save=False):
            key = f"epoch_{self.evaluator.epoch}"
            self.evaluator.results.setdefault(key, {"classification": {}})
            tname = prefix + "_" + data_loader.dataset.name
            if prefix == "val":
                value = self.val_values[self.evaluator.epoch]
            else:
                self.test_calls += 1
                self.test_weight_seen = float(self.model.weight.detach().item())
                value = 0.7
            self.evaluator.results[key]["classification"][tname] = {"top1": value}

    cfg = get_cfg()
    cfg.SOLVER.TOTAL_EPOCH = 3
    cfg.SOLVER.WARMUP_EPOCH = 0
    cfg.SOLVER.PATIENCE = 20
    cfg.SOLVER.OPTIMIZER = "adamw"
    cfg.SOLVER.BASE_LR = 1e-3
    cfg.SOLVER.WEIGHT_DECAY = 0.0
    cfg.SOLVER.LOG_EVERY_N = 99
    cfg.DATA.CLASS_WEIGHTS_TYPE = "none"
    cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH = False
    cfg.OUTPUT_DIR = "/tmp"

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.tensor(0.0)))
    evaluator = TinyEvaluator()
    trainer = TinyTrainer(cfg, model, evaluator, torch.device("cpu"))
    loader = torch.utils.data.DataLoader(TinyDataset(), batch_size=1)
    summary = trainer.train_classifier(loader, loader, loader)

    assert summary["best_epoch"] == 2
    assert summary["best_val_top1"] == 0.8
    assert summary["test_top1"] == 0.7
    assert trainer.test_calls == 1
    assert trainer.test_weight_seen == 2.0
