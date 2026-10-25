from config_helpers import config_overrides

import json
import math

import pytest
import torch
from torch.func import jacrev
from torch.nn.utils import parameters_to_vector

from experiments.baselines import ENGDW
from experiments.pinns import convection_pinn as convection


@pytest.fixture(autouse=True)
def double_cpu():
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def setup_problem(arch="mlp", embed="none"):
    model = convection.build_model(arch, convection.build_embedding(embed), 3, 3)
    batch = convection.sample_batch(7, 4, 3, torch.device("cpu"))
    fn, _ = convection.make_engdw_functions(model, 5, embed != "periodic")
    return model, batch, fn


@pytest.mark.parametrize("arch", ["mlp", "modified", "fused", "fused-modified"])
@pytest.mark.parametrize("embed", ["none", "periodic"])
def test_functional_residual_and_jacobian_match_autograd(arch, embed):
    model, batch, fn = setup_problem(arch, embed)
    params = dict(model.named_parameters())
    actual = fn(params, batch).flatten()
    stacked = convection.stacked_residuals(model, batch, 5, embed != "periodic")
    expected = stacked / math.sqrt(stacked.numel())
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    jd = jacrev(lambda p: fn(p, batch).flatten(), chunk_size=3)(params)
    J = torch.cat([value.flatten(1) for value in jd.values()], dim=1)
    # Independent reverse-mode differentiation through the original PDE path.
    rows = []
    for entry in expected:
        grads = torch.autograd.grad(entry, tuple(params.values()), retain_graph=True)
        rows.append(torch.cat([g.flatten() for g in grads]))
    torch.testing.assert_close(J, torch.stack(rows), atol=1e-11, rtol=1e-10)
    _, jacobian = convection.make_engdw_functions(model, 5, embed != "periodic", 3)
    efficient = jacobian(params, batch)
    for key in params:
        torch.testing.assert_close(efficient[key], jd[key], atol=1e-11, rtol=1e-10)


def test_exact_translation_has_zero_functional_residual():
    class Exact(torch.nn.Module):
        def forward(self, t, x):
            return torch.sin(x - 5 * t)

    model = Exact()
    batch = convection.sample_batch(17, 5, 6, torch.device("cpu"))
    fn, _ = convection.make_engdw_functions(model, 5)
    assert fn({}, batch).abs().max() < 1e-14


@pytest.mark.parametrize("chunk", [None, 1, 4])
@pytest.mark.parametrize("per_sample", [False, True])
@pytest.mark.parametrize("momentum", [0.0, 0.95])
def test_woodbury_step_matches_augmented_least_squares(chunk, per_sample, momentum):
    model, batch, fn = setup_problem()
    params = tuple(model.parameters())
    r = convection.stacked_residuals(model, batch, 5)
    r = r / math.sqrt(r.numel())
    J = torch.stack([
        torch.cat([g.flatten() for g in torch.autograd.grad(v, params, retain_graph=True)])
        for v in r
    ])
    theta = parameters_to_vector(params).detach().clone()
    damping, lr = 1e-3, 0.08
    # Parameter-space solve via QR/SVD, independent of the sample-space Gram.
    A = torch.cat([J, math.sqrt(damping) * torch.eye(theta.numel())])
    previous_direction = torch.randn_like(theta)
    b = torch.cat([r.detach(), math.sqrt(damping) * momentum * previous_direction])
    delta = torch.linalg.lstsq(A, b, driver="gelsd").solution
    _, jacobian = convection.make_engdw_functions(model, 5, chunk_size=chunk)
    opt = ENGDW(model, fn, lr=lr, damping=damping, chunk_size=chunk,
                jacobian_fn=jacobian if per_sample else None, momentum=momentum)
    opt.d_prev.copy_(previous_direction)
    before = r.detach().square().sum().item()
    loss = opt.step(batch)
    assert loss == pytest.approx(before)
    torch.testing.assert_close(parameters_to_vector(params), theta - lr * delta,
                               rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("optimizer", ["gnome", "soap", "adamw", "engdw"])
@pytest.mark.parametrize("embed", ["none", "periodic"])
def test_training_and_metadata(tmp_path, monkeypatch, optimizer, embed):
    torch.set_default_dtype(torch.float32)
    monkeypatch.setattr(convection, "pick_device", lambda: torch.device("cpu"))
    reference = convection.convection_reference
    monkeypatch.setattr(convection, "convection_reference",
                        lambda beta: reference(beta, nt=9, nx=8))
    args = convection.parse_args(config_overrides(convection, [
        "--optimizer", optimizer, "--embed", embed, "--steps", "2",
        "--hidden", "3", "--depth", "3", "--n-pde", "7", "--n-ic", "4",
        "--n-bc", "3", "--engdw-momentum", "0.9", "--engdw-chunk", "3",
        "--log-every", "1", "--runs-dir", str(tmp_path), "--quiet",
    ]))
    path = convection.train(args)
    with open(path) as file:
        records = [json.loads(line) for line in file]
    meta = records[0]["hyperparameters"]
    assert meta["sampling"] == "resample_each_step"
    assert meta["blocks"] == (["pde", "ic", "bc"] if embed == "none" else ["pde", "ic"])
    if optimizer == "engdw":
        assert meta["dtype"] == "torch.float64"
        assert meta["opt.schedule"] == "none"
        assert meta["opt.weight_decay"] == 0
        assert meta["opt.momentum"] == 0.9
        assert meta["opt.residual_scaling"] == "1/sqrt(N_block)"
    train = [r["metrics"] for r in records if r["type"] == "train"]
    assert len(train) == 2
    assert all(math.isfinite(r["loss"]) for r in train)
    assert records[-1]["completed"]
    assert math.isfinite(records[-1]["final_rel_l2"])


@pytest.mark.parametrize("fixed", [False, True])
def test_training_sampling_and_line_search(tmp_path, monkeypatch, fixed):
    # MPS cannot run float64: ENGD-W must fall back to CPU.
    monkeypatch.setattr(convection, "pick_device", lambda: torch.device("mps"))
    reference = convection.convection_reference
    monkeypatch.setattr(convection, "convection_reference",
                        lambda beta: reference(beta, nt=9, nx=8))
    batches, validation = [], []
    original_step, original_terms = ENGDW.step, convection.term_losses

    def step(self, batch):
        batches.append(batch)
        loss = original_step(self, batch)
        assert self.param_groups[0]["lr"] in self.ls_grid
        return loss

    def terms(model, batch, beta):
        validation.append(batch)
        return original_terms(model, batch, beta)

    monkeypatch.setattr(ENGDW, "step", step)
    monkeypatch.setattr(convection, "term_losses", terms)
    args = convection.parse_args(config_overrides(convection, [
        "--optimizer", "engdw", "--steps", "2", "--hidden", "3", "--depth", "3",
        "--n-pde", "7", "--n-ic", "4", "--n-bc", "3", "--engdw-line-search",
        "--log-every", "1", "--runs-dir", str(tmp_path), "--quiet",
    ] + (["--engdw-fixed"] if fixed else [])))
    path = convection.train(args)
    assert (batches[0] is batches[1]) == fixed
    assert all(v is not b for v in validation for b in batches)
    assert not torch.equal(validation[0][0], validation[1][0])
    with open(path) as file:
        meta = json.loads(next(file))["hyperparameters"]
    assert meta["device"] == "cpu"
    assert meta["sampling"] == ("fixed" if fixed else "resample_each_step")
    assert meta["opt.line_search"]


@pytest.mark.parametrize("option,value", [
    ("--engdw-damping", "0"), ("--engdw-damping", "nan"),
    ("--engdw-momentum", "1"), ("--engdw-momentum", "nan"),
    ("--engdw-chunk", "0"), ("--lr", "inf"), ("--n-pde", "0"),
    ("--diagnostics-every", "1"),
])
def test_invalid_engdw_options(option, value):
    with pytest.raises(SystemExit):
        args = convection.parse_args(config_overrides(convection, ["--optimizer", "engdw", option, value]))
        convection.train(args)


def test_fixed_points_require_engdw():
    args = convection.parse_args(config_overrides(convection, ["--optimizer", "gnome", "--engdw-fixed"]))
    with pytest.raises(SystemExit, match="--engdw-fixed requires --optimizer engdw"):
        convection.train(args)
