from config_helpers import config_overrides

import json
import math

import pytest
import torch
from torch.nn.utils import parameters_to_vector

from experiments.baselines import ENGDW
from experiments.pinns import burgers_pinn as burgers


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("chunk", [1, 3, 32])
def test_residual_jacobian_and_parameter_space_update(chunk):
    model = burgers.PINN(hidden=3, depth=3).double()
    batch = burgers.sample_batch(5, 3, 2, torch.device("cpu"), torch.float64)
    residual, jacobian = burgers.make_engdw_functions(model, chunk)
    params = dict(model.named_parameters())
    stacked = burgers.stacked_residuals(model, batch)
    expected = stacked.flatten() / math.sqrt(stacked.numel())
    torch.testing.assert_close(residual(params, batch).flatten(), expected, rtol=1e-12, atol=1e-12)
    J = torch.stack([
        torch.cat([g.flatten() for g in torch.autograd.grad(row, tuple(params.values()), retain_graph=True)])
        for row in expected
    ])
    actual = jacobian(params, batch)
    actual = torch.cat([actual[k].flatten(1) for k in params], 1)
    torch.testing.assert_close(actual, J, rtol=1e-10, atol=1e-11)
    theta = parameters_to_vector(model.parameters()).detach().clone()
    damping, lr = 1e-3, .05
    # Solve in parameter space independently of ENGD-W's sample-space solve.
    matrix = torch.cat((J, math.sqrt(damping) * torch.eye(theta.numel(), dtype=theta.dtype)))
    rhs = torch.cat((expected.detach(), torch.zeros_like(theta)))
    delta = torch.linalg.lstsq(matrix, rhs, driver="gelsd").solution
    opt = ENGDW(model, residual, jacobian_fn=jacobian, lr=lr, damping=damping)
    assert opt.step(batch) == pytest.approx(expected.square().sum().item())
    torch.testing.assert_close(parameters_to_vector(model.parameters()), theta - lr * delta,
                               rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("fixed_flag", [None, "--engdw-fixed", "--fixed-dataset"])
@pytest.mark.parametrize("line_search", [False, True])
def test_training_metadata_sampling_and_final_eval(tmp_path, monkeypatch, fixed_flag, line_search):
    monkeypatch.setattr(burgers, "pick_device", lambda: torch.device("mps"))
    reference = burgers.burgers_reference
    monkeypatch.setattr(burgers, "burgers_reference", lambda: reference(nx=16, nt=5, cache_path=""))
    training, validation = [], []
    original_step, original_terms = ENGDW.step, burgers.term_losses

    def step(self, batch):
        training.append(tuple(t.clone() for t in batch))
        return original_step(self, batch)

    def terms(model, batch):
        validation.append(tuple(t.clone() for t in batch))
        return original_terms(model, batch)

    monkeypatch.setattr(ENGDW, "step", step)
    monkeypatch.setattr(burgers, "term_losses", terms)
    args = burgers.parse_args(config_overrides(burgers, [
        "--optimizer", "engdw", "--hidden", "3", "--depth", "3", "--steps", "3",
        "--n-pde", "5", "--n-ic", "3", "--n-bc", "2", "--log-every", "2",
        "--engdw-chunk", "3", "--engdw-damping", ".001", "--engdw-momentum", ".9",
        "--lr", ".05", "--runs-dir", str(tmp_path), "--no-jaxpi-ref", "--quiet",
    ] + ([fixed_flag] if fixed_flag else []) + (["--engdw-line-search"] if line_search else [])))
    original_dtype = torch.get_default_dtype()
    with open(burgers.train(args)) as file:
        records = [json.loads(line) for line in file]
    assert torch.get_default_dtype() == original_dtype
    cfg = records[0]["hyperparameters"]
    assert cfg["device"] == "cpu" and cfg["dtype"] == "torch.float64"
    assert cfg["opt.schedule"] == "none" and cfg["opt.weight_decay"] == 0
    assert cfg["opt.momentum"] == .9 and cfg["opt.damping"] == .001
    assert cfg["opt.residual_scaling"] == "1/sqrt(N_block)"
    assert cfg["n_pde_aux"] == cfg["n_ic_aux"] == cfg["n_bc_aux"] == 0
    assert cfg["fixed_dataset"] == bool(fixed_flag) and cfg["aux_sampling"] is None
    assert all(torch.equal(a, b) for a, b in zip(training[0], training[1])) == bool(fixed_flag)
    assert all(t.dtype == torch.float64 for t in training[0])
    assert not torch.equal(validation[0][0], training[0][0])
    assert not torch.equal(validation[0][0], validation[1][0])
    metrics = [r["metrics"] for r in records if r["type"] == "train"]
    assert len(metrics) == 3 and all(math.isfinite(m["loss"]) for m in metrics)
    if line_search:
        assert all(m["lr"] in cfg["opt.ls_grid"] for m in metrics)
    else:
        assert all(m["lr"] == .05 for m in metrics)
    assert [r["step"] for r in records if r["type"] == "val"] == [2, 3]
    assert records[-1]["completed"] and math.isfinite(records[-1]["final_rel_l2"])


@pytest.mark.parametrize("optimizer", ["gnome", "soap", "adamw"])
@pytest.mark.parametrize("fixed", [False, True])
def test_existing_optimizers_and_fixed_sampling(tmp_path, monkeypatch, optimizer, fixed):
    monkeypatch.setattr(burgers, "pick_device", lambda: torch.device("cpu"))
    reference = burgers.burgers_reference
    monkeypatch.setattr(burgers, "burgers_reference", lambda: reference(nx=16, nt=5, cache_path=""))
    calls = []
    residual = burgers.stacked_residuals

    def capture(model, batch):
        calls.append(tuple(t.clone() for t in batch))
        return residual(model, batch)

    monkeypatch.setattr(burgers, "stacked_residuals", capture)
    args = burgers.parse_args(config_overrides(burgers, [
        "--optimizer", optimizer, "--hidden", "3", "--depth", "3", "--steps", "2",
        "--n-pde", "12", "--n-ic", "5", "--n-bc", "4", "--log-every", "0",
        "--warmup-steps", "0", "--runs-dir", str(tmp_path), "--no-jaxpi-ref", "--quiet",
    ] + (["--fixed-dataset"] if fixed else [])))
    with open(burgers.train(args)) as file:
        records = [json.loads(line) for line in file]
    main = [b for b in calls if len(b[0]) == 12]
    assert len(main) == 2
    assert all(torch.equal(a, b) for a, b in zip(*main)) == fixed
    if optimizer == "gnome" and fixed:
        aux = [b for b in calls if len(b[0]) == 8]
        assert len(aux) == 2
        for child, parent in zip(aux, main):
            groups = [(torch.cat(child[:2], 1), torch.cat(parent[:2], 1)),
                      (child[2], parent[2]), (child[3], parent[3])]
            for subset, full in groups:
                assert (subset[:, None] == full[None]).all(-1).any(-1).all()
    assert records[-1]["completed"] and math.isfinite(records[-1]["final_rel_l2"])
    assert [r["step"] for r in records if r["type"] == "val"] == [2]


@pytest.mark.parametrize("option,value", [
    ("--engdw-damping", "0"), ("--engdw-damping", "nan"),
    ("--engdw-momentum", "1"), ("--engdw-momentum", "nan"),
    ("--engdw-chunk", "0"), ("--lr", "inf"), ("--n-ic", "0"),
    ("--diagnostics-every", "1"),
])
def test_invalid_engdw_options(option, value):
    with pytest.raises(SystemExit):
        burgers.parse_args(config_overrides(burgers, ["--optimizer", "engdw", option, value]))


def test_fixed_alias_restricted_to_engdw():
    with pytest.raises(SystemExit):
        burgers.parse_args(config_overrides(burgers, ["--optimizer", "adamw", "--engdw-fixed"]))
