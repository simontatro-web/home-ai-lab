from pathlib import Path

import pytest
import torch
import yaml

from model.transformer import Transformer

CONFIGS = Path(__file__).resolve().parents[1] / "training" / "configs"
# Exact totals for a 50k vocabulary with tied embeddings.
EXPECTED = {
    "model1_44M.yaml": 44_481_024,
    "model2_71M.yaml": 71_332_480,
    "model3_123M.yaml": 123_353_856,
    "model4_236M.yaml": 235_783_168,
    "model4_236M_raw.yaml": 235_783_168,
    "model5_457M.yaml": 457_268_480,
}


def tiny(**kw):
    args = dict(vocab_size=64, n_layer=2, d_model=32, n_heads=4, ctx=16)
    return Transformer(**{**args, **kw})


@pytest.mark.parametrize("name,count", EXPECTED.items())
def test_config_param_counts(name, count):
    m = yaml.safe_load((CONFIGS / name).read_text())["model"]
    with torch.device("meta"):
        net = Transformer(50_000, m["n_layer"], m["d_model"], m["n_heads"], m.get("n_kv_heads"), m["ctx"])
    assert net.num_params() == count


def test_weights_are_tied():
    net = tiny()
    assert net.head.weight is net.wte.weight


@pytest.mark.parametrize("kv", [4, 2])
def test_attention_is_causal(kv):
    net = tiny(n_kv_heads=kv).eval()
    a = torch.randint(0, 64, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 64  # change only the future
    with torch.no_grad():
        la, _ = net(a)
        lb, _ = net(b)
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-5)
    assert not torch.allclose(la[0, 8:], lb[0, 8:], atol=1e-5)


def test_output_depends_on_token_order():
    # With no position information, swapping two earlier tokens would not
    # change the output at the last position. RoPE makes it matter.
    net = tiny(n_layer=1).eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[2, 1, 3, 4, 5, 6]])
    with torch.no_grad():
        la, _ = net(a)
        lb, _ = net(b)
    assert not torch.allclose(la[0, -1], lb[0, -1], atol=1e-5)


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    net = tiny()
    x = torch.randint(0, 64, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(400):
        _, loss = net(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_rejects_too_long_input():
    with pytest.raises(AssertionError):
        tiny()(torch.zeros((1, 17), dtype=torch.long))
