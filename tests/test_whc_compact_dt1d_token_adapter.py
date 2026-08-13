import torch

from src.configs.config import get_cfg
from src.configs import vit_configs
from src.models.vit_adapter.vit import ADPT_Block
from src.models.vit_adapter.whc_compact_dt1d_adapter import (
    WHCCompactDT1DAdapter,
    WHCCompactDT1DTokenAdapter,
)


def test_final_whc_defaults():
    m = WHCCompactDT1DAdapter(32)
    assert m.axis == "hw"
    assert m.group_size == 16
    assert m.active_offsets == (1, 2, 4)
    assert m.whc_p == 2
    assert m.effective_kernel_size == 13
    assert m.project_l1 is True
    assert m.gate_mode == "fixed"
    assert not isinstance(m.gate, torch.nn.Parameter)
    assert torch.isclose(m.gate, torch.tensor(0.01))
    assert m.whc_lambda_mode == "learned"
    assert m.whc_lambda_scope == "block"
    assert m.use_pointwise is False


def test_final_whc_joint_l1_projection():
    torch.manual_seed(1)
    m = WHCCompactDT1DAdapter(32)
    with torch.no_grad():
        m.base_coefficients.normal_()
        m.detail_coefficients.normal_()
        m.whc_theta.fill_(0.7)
    k = m.build_kernels(torch.device("cpu"), torch.float32)
    mass = k.squeeze(2).abs().sum(-1).sum(0)
    assert torch.all(mass <= 1.000001)


def test_whc_token_adapter_preserves_class_token_and_backpropagates():
    torch.manual_seed(2)
    m = WHCCompactDT1DTokenAdapter(embed_dim=16, grid_size=(4, 4))
    x = torch.randn(2, 17, 16, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y[:, :1], x[:, :1])
    y.square().mean().backward()
    assert x.grad is not None
    assert m.spatial_adapter.whc_theta.grad is not None
    assert torch.isfinite(m.spatial_adapter.whc_theta.grad).all()


def test_whc_initial_lambda_zero_matches_compact_l1_fixed_gate():
    torch.manual_seed(3)
    whc = WHCCompactDT1DAdapter(16)
    x = torch.randn(2, 16, 8, 8)
    full = whc.build_unprojected_kernels(torch.device("cpu"), torch.float32)
    assert full.shape[-1] == 13
    # lambda=0 means the K9 base is centered exactly inside K13.
    base17 = super(WHCCompactDT1DAdapter, whc).build_kernels(
        torch.device("cpu"), torch.float32, project=False
    )
    base9 = base17[..., 4:13]
    assert torch.allclose(full[..., 2:11], base9)
    assert torch.count_nonzero(full[..., :2]).item() == 0
    assert torch.count_nonzero(full[..., -2:]).item() == 0
    assert torch.isfinite(whc(x)).all()


def test_vit_block_builds_final_whc_adapter():
    cfg = get_cfg()
    cfg.MODEL.ADAPTER.NAME = "WHC_DT1D"
    block = ADPT_Block(
        vit_configs.get_b16_config(),
        False,
        cfg.MODEL.ADAPTER,
        grid_size=(14, 14),
    )
    m = block.token_adapter.spatial_adapter
    assert isinstance(m, WHCCompactDT1DAdapter)
    assert m.whc_p == 2
    assert m.gate_mode == "fixed"
    assert m.effective_kernel_size == 13


def test_whc_yaml_contracts():
    cfg = get_cfg()
    assert cfg.MODEL.ADAPTER.WHC_DT1D.ACTIVE_OFFSETS == "1,2,4"
    assert cfg.MODEL.ADAPTER.WHC_DT1D.P == 2
    assert cfg.MODEL.ADAPTER.WHC_DT1D.GATE_MODE == "fixed"
    assert cfg.MODEL.ADAPTER.WHC_DT1D.GATE_INIT == 0.01
