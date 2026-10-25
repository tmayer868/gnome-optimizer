from config_helpers import config_overrides

import ast
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments.transformers import modded_nanogpt as bench
from experiments.transformers.fineweb_data import TokenStream, shard_info
from experiments.transformers.modded_nanogpt_model import GPT


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def write_shard(path, tokens):
    header = np.zeros(256, dtype="<i4")
    header[:3] = [20240520, 1, len(tokens)]
    with path.open("wb") as f:
        header.tofile(f)
        np.asarray(tokens, dtype="<u2").tofile(f)


def tiny_args(optimizer="adamw", *extra):
    return bench.parse_args(config_overrides(bench, [
        "--preset", "dev", "--synthetic", "--optimizer", optimizer, "--device", "cpu",
        "--n-layer", "1", "--n-embd", "16", "--head-dim", "8", "--vocab-size", "32",
        "--seq-len", "4", "--batch-tokens", "20", "--val-tokens", "28", "--steps", "3",
        "--microbatch-size", "2", "--precondition-frequency", "1", *extra,
    ]))


def test_defaults_and_schedule_match_reference():
    args = bench.parse_args(config_overrides(bench, ["--optimizer", "muon"]))
    assert all(getattr(args, key) == value for key, value in bench.REFERENCE.items())
    assert (args.steps, args.lr, args.weight_decay) == (3250, 0.025, 0.05)
    assert (args.beta1, args.beta2, args.eps) == (0.8, 0.95, 1e-10)
    assert (args.embed_lr, args.head_lr, args.scalar_lr) == (0.7, 0.004, 0.015)
    assert args.head_init_std == 0
    assert args.lr_schedule == "linear"
    for step in (0, 974, 975, 2000, 3249):
        expected = 1 if step / 3250 < 0.3 else (1 - step / 3250) / 0.7
        assert bench.schedule_multiplier(step, 3250, 0.7) == pytest.approx(expected)


@pytest.mark.parametrize("floor", [0.0, 0.1, 1.0])
def test_cosine_schedule_warmup_and_endpoints(floor):
    def multiplier(step, steps=7, warmup=3, cooldown=0.7):
        return bench.schedule_multiplier(step, steps, cooldown, warmup,
                                         lr_schedule="cosine", min_lr_frac=floor)

    values = [multiplier(step) for step in range(7)]
    assert values[:3] == pytest.approx([1 / 3, 2 / 3, 1])
    assert values[4] == pytest.approx((1 + floor) / 2)
    assert values[-1] == pytest.approx(floor)
    assert all(a >= b for a, b in zip(values[2:], values[3:]))
    assert multiplier(3, cooldown=0) == multiplier(3, cooldown=1)
    assert multiplier(0, steps=5, warmup=0) == pytest.approx(1)
    assert multiplier(2, steps=5, warmup=0) == pytest.approx((1 + floor) / 2)
    assert multiplier(4, steps=5, warmup=0) == pytest.approx(floor)
    assert multiplier(6, warmup=6) == pytest.approx(floor)
    assert multiplier(0, steps=1, warmup=0) == pytest.approx(floor)
    assert multiplier(100) == pytest.approx(floor)


@pytest.mark.parametrize("optimizer,embedding_optimizer", [
    ("gnome_hutchinson", "same"), ("gnome_hutchinson", "adamw"), ("muon", "same"),
])
def test_cosine_schedule_applied_to_groups_and_logged(tmp_path, optimizer, embedding_optimizer):
    args = tiny_args(optimizer, "--runs-dir", str(tmp_path), "--quiet", "--save-checkpoint",
                     "--embedding-optimizer", embedding_optimizer, "--steps", "7",
                     "--warmup-steps", "3", "--lr-schedule", "cosine", "--min-lr-frac", "0.1")
    path = bench.run(args)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    hp = records[0]["hyperparameters"]
    assert (hp["lr_schedule"], hp["min_lr_frac"]) == ("cosine", 0.1)
    train = [r["metrics"] for r in records if r["type"] == "train"]
    expected = [1 / 3, 2 / 3, 1, 0.1 + 0.45 * (1 + math.sqrt(0.5)),
                0.55, 0.1 + 0.45 * (1 - math.sqrt(0.5)), 0.1]
    assert [r["lr"] for r in train] == pytest.approx([args.lr * v for v in expected])
    if embedding_optimizer == "adamw":
        assert [r["embedding_lr"] for r in train] == pytest.approx([args.embed_lr * v for v in expected])
    saved = torch.load(records[-1]["checkpoint"], weights_only=True)
    for opt in saved["optimizers"]:
        for group in opt["param_groups"]:
            assert group["lr"] == pytest.approx(group["initial_lr"] * 0.1)


def test_model_matches_upstream_forward_and_gradients():
    # Execute architecture definitions only: no CUDA setup, training, or imports
    # with side effects from the preserved upstream script.
    path = Path(bench.__file__).with_name("modded_nanogpt_reference") / "train_gpt_simple.py"
    tree = ast.parse(path.read_text())
    names = {"RMSNorm", "Linear", "Rotary", "CausalSelfAttention", "MLP", "Block", "GPT"}
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    scope = dict(torch=torch, Tensor=torch.Tensor, nn=nn, F=F)
    exec(compile(tree, str(path), "exec"), scope)
    torch.manual_seed(3)
    model = GPT(vocab_size=48, num_layers=2, model_dim=128)
    # Compare architecture independently of initialization by loading the
    # exact same parameters into the preserved upstream architecture.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "proj" in name:
                param.normal_(std=0.03)
    reference = scope["GPT"](48, 2, 128).float()
    reference.load_state_dict(model.state_dict(), strict=True)
    x, y = torch.randint(48, (2, 7)), torch.randint(48, (2, 7))
    logits = model(x)
    actual = F.cross_entropy(logits.reshape(-1, 48), y.reshape(-1), reduction="sum")
    expected = reference(x, y)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.backward()
    expected.backward()
    for param, ref in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, ref.grad, atol=1e-6, rtol=1e-5)
    assert model.embed.weight is not model.proj.weight
    # Future tokens cannot affect earlier logits.
    altered = x.clone()
    altered[:, -1] = (altered[:, -1] + 1) % 48
    torch.testing.assert_close(model(altered)[:, :-1], logits[:, :-1])


def test_reference_dimensions_and_initialization():
    with torch.device("meta"):
        model = GPT()
    assert len(model.blocks) == 12
    assert model.embed.weight.shape == model.proj.weight.shape == (50304, 768)
    assert model.blocks[0].attn.num_heads == 6
    assert model.blocks[0].mlp.fc.weight.shape == (3072, 768)
    torch.manual_seed(11)
    small = GPT(512, 3, 128, 32)
    for name, param in small.named_parameters():
        if name.endswith("bias"):
            assert torch.count_nonzero(param) == 0
        if name.endswith("gains"):
            assert torch.equal(param, torch.ones_like(param))
        if name.endswith("weight"):
            if "proj" in name:
                assert torch.count_nonzero(param) == 0
                continue
            expected_std = 1 if "embed" in name else math.sqrt(0.33 / param.size(-1))
            assert param.std().item() == pytest.approx(expected_std, rel=0.03)
            assert abs(param.mean().item()) < expected_std * 0.03
    assert small.embed.weight is not small.proj.weight


def test_initialization_matches_saved_reference():
    model = GPT(128, 2, 64, 32)
    reference = deepcopy(model)
    path = Path(bench.__file__).with_name("modded_nanogpt_reference") / "train_gpt_simple.py"
    tree = ast.parse(path.read_text())
    init_loop = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.For) and ast.unparse(node.iter) == "model.named_parameters()")
    torch.manual_seed(73)
    model.reset_parameters()
    torch.manual_seed(73)
    exec(compile(ast.Module(body=[init_loop], type_ignores=[]), str(path), "exec"),
         {"model": reference})
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_zero_initialization_starts_uniform_and_gnome_opens_gradient_paths():
    torch.manual_seed(10)
    args = tiny_args("gnome_hutchinson")
    model = GPT(32, 1, 16, 8)
    assert torch.count_nonzero(model.proj.weight) == 0
    assert torch.count_nonzero(model.proj.bias) == 0
    x = torch.randint(32, (5, 4))
    logits = model(x)
    assert torch.count_nonzero(logits) == 0
    loss = F.cross_entropy(logits.reshape(-1, 32), ((x + 1) % 32).reshape(-1))
    assert loss.item() == pytest.approx(math.log(32))
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        if name not in ("proj.weight", "proj.bias"):
            assert torch.count_nonzero(param.grad) == 0, name
    opts = bench.build_optimizers(model, args, torch.device("cpu"))
    for _ in range(5):
        bench.train_step(model, opts, (x, (x + 1) % 32), args,
                         torch.device("cpu"), torch.Generator())
        assert all(torch.isfinite(p).all() for p in model.parameters())
    assert torch.count_nonzero(model.proj.weight) > 0
    assert torch.count_nonzero(model.blocks[0].attn.proj.weight) > 0
    assert torch.count_nonzero(model.blocks[0].mlp.proj.weight) > 0


def test_head_override_leaves_residual_projections_zero():
    model = GPT(512, 1, 64, 32, head_init_std=0.02)
    assert model.proj.weight.std().item() == pytest.approx(0.02, rel=0.03)
    assert torch.count_nonzero(model.blocks[0].attn.proj.weight) == 0
    assert torch.count_nonzero(model.blocks[0].mlp.proj.weight) == 0


def test_fp32_embedding_preserves_bf16_forward_but_accumulates_small_updates():
    torch.manual_seed(4)
    reference = GPT(32, 1, 16, 8, torch.bfloat16, head_init_std=0.02)
    torch.manual_seed(4)
    master = GPT(32, 1, 16, 8, torch.bfloat16, head_init_std=0.02, fp32_embedding=True)
    assert reference.embed.weight.dtype == torch.bfloat16
    assert master.embed.weight.dtype == torch.float32
    torch.testing.assert_close(master.embed.weight.bfloat16(), reference.embed.weight, rtol=0, atol=0)
    x = torch.randint(32, (2, 4))
    torch.testing.assert_close(master(x), reference(x), rtol=0, atol=0)
    # Forty sub-ULP updates disappear individually in BF16, but their sum is
    # representable in BF16 after accumulating in the FP32 master parameter.
    with torch.no_grad():
        reference.embed.weight[0].fill_(1)
        master.embed.weight[0].fill_(1)
        for _ in range(40):
            reference.embed.weight[0].add_(1e-4)
            master.embed.weight[0].add_(1e-4)
    assert torch.equal(reference.embed.weight[0], torch.ones(16, dtype=torch.bfloat16))
    assert (master.embed.weight[0].bfloat16() > 1).all()
    args = tiny_args("gnome_hutchinson")
    opts = bench.build_optimizers(master, args, torch.device("cpu"))
    for _ in range(3):
        bench.train_step(master, opts, (x, (x + 1) % 32), args,
                         torch.device("cpu"), torch.Generator())
    assert opts[0].state[master.embed.weight]["grad_m"].dtype == torch.float32
    assert opts[0].state[master.embed.weight]["gnd_m"].dtype == torch.float32


def test_shard_boundary_order_capacity_and_reset(tmp_path):
    write_shard(tmp_path / "train_01.bin", range(18))
    write_shard(tmp_path / "train_02.bin", range(20, 38))
    stream = TokenStream(tmp_path / "train_*.bin", 8, 4, 64)
    assert stream.capacity == 4
    for start in (0, 8, 20, 28):
        x, y = stream.next_batch()
        assert x.flatten().tolist() == list(range(start, start + 8))
        assert y.flatten().tolist() == list(range(start + 1, start + 9))
    with pytest.raises(RuntimeError, match="exhausted"):
        stream.next_batch()
    stream.reset()
    assert stream.next_batch()[0][0, 0] == 0
    # Preserve the upstream >= boundary condition, even for an exact fit.
    write_shard(tmp_path / "exact.bin", range(9))
    assert TokenStream(tmp_path / "exact.bin", 8, 4).capacity == 0


def test_invalid_shards_and_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenStream(tmp_path / "missing*", 8, 4)
    path = tmp_path / "bad.bin"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="header"):
        shard_info(path)
    write_shard(path, range(12))
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="size"):
        shard_info(path)
    write_shard(path, range(12))
    with pytest.raises(ValueError, match="vocabulary"):
        TokenStream(path, 8, 4, vocab_size=4).next_batch()


@pytest.mark.parametrize("optimizer,embedding_optimizer", [
    ("adamw", "same"), ("gnome_hutchinson", "same"), ("gnome_fisher", "same"), ("muon", "same"),
    ("gnome_hutchinson", "adamw"), ("gnome_fisher", "adamw"),
])
def test_microbatch_accumulation_matches_full_batch(optimizer, embedding_optimizer):
    args = tiny_args(optimizer, "--embedding-optimizer", embedding_optimizer)
    torch.manual_seed(10)
    model = GPT(args.vocab_size, args.n_layer, args.n_embd, args.head_dim)
    full = deepcopy(model)
    full_args = deepcopy(args)
    full_args.microbatch_size = 5  # uneven 2+2+1 accumulation vs whole batch
    opts = bench.build_optimizers(model, args, torch.device("cpu"))
    full_opts = bench.build_optimizers(full, full_args, torch.device("cpu"))
    data = torch.randint(args.vocab_size, (21,))
    batch = (data[:-1].view(5, 4), data[1:].view(5, 4))
    for step in range(3):
        torch.manual_seed(step)
        loss = bench.train_step(model, opts, batch, args, torch.device("cpu"), torch.Generator().manual_seed(step))
        torch.manual_seed(step)
        expected = bench.train_step(full, full_opts, batch, full_args, torch.device("cpu"),
                                    torch.Generator().manual_seed(step))
        assert loss == pytest.approx(expected, rel=1e-5)
    # Curvature eigenspaces may be numerically different with degenerate factors;
    # tolerate FP32 accumulation error propagated through the optimizer.
    for p, q in zip(model.parameters(), full.parameters()):
        assert torch.isfinite(p).all()
        torch.testing.assert_close(p, q, atol=3e-4, rtol=3e-3)
    assert torch.count_nonzero(model.proj.weight) > 0


def test_soap_receives_same_accumulated_gradient():
    # SOAP's basis is not unique for rank-deficient early gradients. Verify
    # the gradient input directly rather than requiring identical eigenvectors.
    args = tiny_args("soap")
    model = GPT(args.vocab_size, args.n_layer, args.n_embd, args.head_dim)
    full = deepcopy(model)
    data = torch.randint(args.vocab_size, (21,))
    batch = (data[:-1].view(5, 4), data[1:].view(5, 4))
    for candidate, microbatch in ((model, 2), (full, 5)):
        opts = bench.build_optimizers(candidate, args, torch.device("cpu"))
        opts[0].step = lambda: None
        args.microbatch_size = microbatch
        bench.train_step(candidate, opts, batch, args, torch.device("cpu"), torch.Generator())
    for p, q in zip(model.parameters(), full.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5)


def test_evaluation_weights_tokens_and_restores_mode():
    class Lookup(nn.Module):
        def forward(self, x):
            return F.one_hot(x, 4).float() * 3

    x = torch.tensor([[0, 1], [2, 3], [0, 2]])
    y = torch.tensor([[0, 1], [1, 3], [2, 1]])
    model = Lookup().train()
    loss = bench.evaluate(model, (x, y), 2, torch.device("cpu"))
    expected = F.cross_entropy(model(x).reshape(-1, 4), y.reshape(-1)).item()
    assert loss == pytest.approx(expected)
    assert model.training


@pytest.mark.parametrize("optimizer,embedding_optimizer", [
    ("adamw", "same"), ("soap", "same"), ("gnome_hutchinson", "same"), ("gnome_fisher", "same"),
    ("muon", "same"), ("gnome_hutchinson", "adamw"), ("gnome_fisher", "adamw"),
])
def test_end_to_end_logs_and_checkpoint(tmp_path, optimizer, embedding_optimizer):
    args = tiny_args(optimizer, "--runs-dir", str(tmp_path), "--save-checkpoint", "--quiet",
                     "--embedding-optimizer", embedding_optimizer, "--embed-lr", "0.002")
    path = bench.run(args)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    hp = records[0]["hyperparameters"]
    assert hp["initialization"] == "reference"
    assert hp["head_init_std"] == 0
    assert hp["residual_init_std"] == 0
    assert hp["reference_initialization_matches"]
    assert not hp["reference_configuration_matches"]
    assert hp["extra_curvature_pass"] == optimizer.startswith("gnome")
    assert hp["microbatches_per_step"] == 3
    assert Path(hp["source_bundle"]).is_file()
    assert hp["source_sha256"]
    assert "experiments/configs/transformers/modded_nanogpt.py" in hp["source_sha256"]
    assert "experiments/common/config_schema.py" in hp["source_sha256"]
    train = [r for r in records if r["type"] == "train"]
    assert [r["step"] for r in train] == [1, 2, 3]
    assert train[-1]["metrics"]["tokens_seen"] == 60
    assert records[-1]["completed"]
    assert records[-1]["completed_steps"] == 3
    assert records[-1]["training_seconds"] > 0
    checkpoint = torch.load(records[-1]["checkpoint"], weights_only=True)
    assert checkpoint["step"] == 3
    if embedding_optimizer == "adamw":
        assert hp["embedding_optimizer"] == "adamw"
        for row in train:
            multiplier = bench.schedule_multiplier(row["step"] - 1, args.steps, args.cooldown_frac)
            assert row["metrics"]["embedding_lr"] == pytest.approx(0.002 * multiplier)
        assert len(checkpoint["optimizers"]) == 1
        groups = checkpoint["optimizers"][0]["param_groups"]
        assert [g["update_rule"] for g in groups] == ["gnome", "adamw"]
        embed_state = checkpoint["optimizers"][0]["state"][groups[1]["params"][0]]
        assert set(embed_state) == {"step", "exp_avg", "exp_avg_sq"}
        assert embed_state["step"] == 3


def test_gnome_embedding_adamw_groups_and_first_update():
    args = tiny_args("gnome_hutchinson", "--embedding-optimizer", "adamw",
                     "--embed-lr", "0.002", "--embed-beta1", "0.8", "--embed-beta2", "0.95",
                     "--embed-eps", "1e-7", "--embed-weight-decay", "0.1")
    model = GPT(args.vocab_size, args.n_layer, args.n_embd, args.head_dim, head_init_std=0.02)
    opts = bench.build_optimizers(model, args, torch.device("cpu"))
    assert len(opts) == 1
    gnome, adam = opts[0].param_groups
    assert adam["params"] == [model.embed.weight]
    assert all(p is not model.embed.weight for p in gnome["params"])
    assert any(p is model.proj.weight for p in gnome["params"])
    assert (adam["lr"], adam["betas"], adam["eps"], adam["weight_decay"]) == (
        0.002, (0.8, 0.95), 1e-7, 0.1)
    original = deepcopy(model.state_dict())
    x, y = torch.randint(32, (5, 4)), torch.randint(32, (5, 4))
    bench.train_step(model, opts, (x, y), args, torch.device("cpu"), torch.Generator())
    assert not torch.equal(model.embed.weight, original["embed.weight"])
    for name, p in model.named_parameters():
        if name != "embed.weight":
            torch.testing.assert_close(p, original[name], atol=0, rtol=0)


def test_embedding_override_requires_gnome():
    with pytest.raises(SystemExit):
        tiny_args("soap", "--embedding-optimizer", "adamw")


def test_real_shard_run_and_exhaustion_preflight(tmp_path):
    write_shard(tmp_path / "fineweb_train_000001.bin", np.arange(90) % 32)
    write_shard(tmp_path / "fineweb_val_000000.bin", np.arange(32) % 32)
    args = tiny_args("adamw", "--data-dir", str(tmp_path), "--runs-dir", str(tmp_path))
    args.synthetic = False
    assert bench.run(args).is_file()
    args.steps = 5
    with pytest.raises(ValueError, match="Only 4 complete"):
        bench.run(args)


def test_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert bench.select_device("auto").type == "cuda"
    assert bench.select_dtype("auto", torch.device("cuda")) == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bench.select_device("auto").type == "mps"
    assert bench.select_dtype("auto", torch.device("mps")) == torch.float32
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert bench.select_device("auto").type == "cpu"
    for name in ("mps", "cuda"):
        with pytest.raises(ValueError, match="unavailable"):
            bench.select_device(name)
    with pytest.raises(ValueError, match="BF16"):
        bench.select_dtype("bfloat16", torch.device("mps"))


@pytest.mark.parametrize("argv", [
    ["--batch-tokens", "13"], ["--seq-len", "0"], ["--synthetic"],
    ["--head-dim", "127"], ["--steps", "0"], ["--val-every", "0"],
    ["--microbatch-size", "0"], ["--download-only"], ["--download-shards", "104"],
    ["--beta1", "nan"], ["--lr", "inf"], ["--cooldown-frac", "1.1"],
    ["--head-init-std", "-0.02"], ["--head-init-std", "nan"],
    ["--min-lr-frac", "-0.1"], ["--min-lr-frac", "1.1"],
    ["--min-lr-frac", "nan"], ["--min-lr-frac", "inf"], ["--lr-schedule", "unknown"],
])
def test_invalid_arguments(argv):
    with pytest.raises(SystemExit):
        bench.parse_args(config_overrides(bench, argv))
