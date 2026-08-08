import torch

from src.models.vit_adapter.dt1d_adapter import DT1DAdapter, DT1DTokenAdapter


def test_final_defaults_match_paper_architecture():
    m = DT1DAdapter(32)
    assert m.axis == "hw"
    assert m.group_size == 16
    assert m.active_offsets == (1, 2, 4, 8)
    assert m.detail_components == "offset4"
    assert m.detail_basis == "orth"
    assert m.project_l1 is True
    assert m.use_pointwise is False
    assert m.convolution_calls_per_forward == 2
    assert torch.isclose(m.gate.detach(), torch.tensor(0.01))


def test_kaggle_active_offsets_string_contract():
    m = DT1DAdapter(32, active_offsets="1,2,4,8")
    assert m.active_offsets == (1, 2, 4, 8)


def test_joint_l1_projection():
    torch.manual_seed(0)
    m = DT1DAdapter(24)
    with torch.no_grad():
        m.base_coefficients.normal_()
        m.detail_coefficients.normal_()
    k = m.build_kernels(torch.device("cpu"), torch.float32)
    mass = k.squeeze(2).abs().sum(-1).sum(0)
    assert torch.all(mass <= 1.000001)


def test_token_adapter_preserves_prefix_token_and_shape():
    torch.manual_seed(0)
    m = DT1DTokenAdapter(embed_dim=16, grid_size=(4, 4))
    x = torch.randn(2, 17, 16, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y[:, :1], x[:, :1])
    y.mean().backward()
    assert x.grad is not None


def test_channel_contrast_zero_mean_per_full_group():
    m = DT1DAdapter(32, group_size=16)
    a = m.channel_contrast
    assert abs(float(a[:16].sum())) < 1e-6
    assert abs(float(a[16:32].sum())) < 1e-6


def test_kaggle_yaml_schema_contracts(tmp_path):
    import yaml
    from src.configs.config import get_cfg

    cfg_path = tmp_path / "kaggle_contract.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "MODEL": {
            "ADAPTER": {
                "REDUCTION_FACTOR": 16,
                "DT1D": {"ACTIVE_OFFSETS": "1,2,4,8"},
            }
        }
    }), encoding="utf-8")
    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_path))
    assert cfg.MODEL.ADAPTER.REDUCTION_FACTOR == 16
    assert cfg.MODEL.ADAPTER.DT1D.ACTIVE_OFFSETS == "1,2,4,8"


def test_kaggle_pfeiffer_canonical_reduction_factor():
    from src.configs.config import get_cfg
    from src.configs import vit_configs
    from src.models.vit_adapter.vit import ADPT_Block

    cfg = get_cfg()
    cfg.MODEL.ADAPTER.NAME = "Pfeiffer"
    cfg.MODEL.ADAPTER.REDUCTION_FACTOR = 16
    block = ADPT_Block(vit_configs.get_b16_config(), False, cfg.MODEL.ADAPTER, grid_size=(14, 14))
    assert block.adapter_downsample.out_features == 48


def test_kaggle_modern_timm_import_stack():
    # This import traverses the same eager backbone imports that previously
    # failed on Kaggle with timm.models.layers.helpers.
    from src.models import build_vit_backbone
    assert callable(build_vit_backbone.build_vit_sup_models)
