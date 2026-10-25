from config_helpers import config_overrides

import json
import math

import pytest
import torch
from torch.nn.utils import parameters_to_vector

from experiments.baselines import ENGDW
from experiments.pinns import poisson_pinn as poisson


@pytest.fixture(autouse=True)
def double_cpu():
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


@pytest.mark.parametrize("arch", ["mlp", "fused"])
def test_residual_jacobian_and_update_match_original_autograd(arch):
    assert poisson.ENGDW is ENGDW
    model = poisson.build_model(arch, hidden=3, depth=4, fuse_every=2)
    batch = poisson.sample_batch(5, 2, torch.device("cpu"))
    fn, jacobian = poisson.make_engdw_functions(model, chunk_size=3)
    params = dict(model.named_parameters())
    stacked = poisson.stacked_residuals(model, batch)
    expected = stacked / math.sqrt(stacked.numel())
    torch.testing.assert_close(fn(params, batch).flatten(), expected, rtol=1e-12, atol=1e-12)
    J = torch.stack([
        torch.cat([g.flatten() for g in torch.autograd.grad(r, tuple(params.values()), retain_graph=True)])
        for r in expected
    ])
    actual = jacobian(params, batch)
    actual = torch.cat([actual[k].flatten(1) for k in params], 1)
    torch.testing.assert_close(actual, J, rtol=1e-10, atol=1e-11)
    theta = parameters_to_vector(model.parameters()).detach().clone()
    damping, lr = 1e-3, 0.05
    # Independent parameter-space least squares vs shared sample-space solve.
    matrix = torch.cat((J, math.sqrt(damping)*torch.eye(theta.numel())))
    rhs = torch.cat((expected.detach(), torch.zeros_like(theta)))
    delta = torch.linalg.lstsq(matrix, rhs, driver="gelsd").solution
    opt = ENGDW(model, fn, jacobian_fn=jacobian, damping=damping, lr=lr)
    assert opt.step(batch) == pytest.approx(expected.detach().square().sum().item())
    torch.testing.assert_close(parameters_to_vector(model.parameters()), theta-lr*delta,
                               rtol=1e-9, atol=1e-10)


def test_exact_solution_has_zero_residual():
    class Exact(torch.nn.Module):
        def forward(self, x, y):
            return torch.sin(math.pi*x)*torch.sin(math.pi*y)
    fn, _ = poisson.make_engdw_functions(Exact(), chunk_size=4)
    assert fn({}, poisson.sample_batch(11, 3, torch.device("cpu"))).abs().max() < 1e-13


@pytest.mark.parametrize("arch", ["mlp", "fused"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("line_search", [False, True])
def test_engdw_training_sampling_and_metadata(tmp_path, monkeypatch, arch, fixed, line_search):
    torch.set_default_dtype(torch.float32)
    # Exercise automatic MPS -> CPU fallback, without needing MPS hardware.
    monkeypatch.setattr(poisson, "pick_device", lambda: torch.device("mps"))
    reference = poisson.poisson_reference
    monkeypatch.setattr(poisson, "poisson_reference", lambda: reference(9, 8))
    train_batches, val_batches = [], []
    original_step, original_terms = ENGDW.step, poisson.term_losses
    def step(self, batch):
        train_batches.append(tuple(t.clone() for t in batch))
        return original_step(self, batch)
    def terms(model, batch):
        val_batches.append(tuple(t.clone() for t in batch))
        return original_terms(model, batch)
    monkeypatch.setattr(ENGDW, "step", step)
    monkeypatch.setattr(poisson, "term_losses", terms)
    args = poisson.parse_args(config_overrides(poisson, [
        "--optimizer", "engdw", "--arch", arch, "--hidden", "3", "--depth", "3",
        "--steps", "3", "--n-pde", "5", "--n-bc-per-edge", "2", "--log-every", "2",
        "--engdw-chunk", "3", "--engdw-damping", "0.001", "--engdw-momentum", "0.9",
        "--lr", "0.05", "--runs-dir", str(tmp_path), "--quiet",
    ] + (["--engdw-fixed"] if fixed else []) + (["--engdw-line-search"] if line_search else [])))
    path = poisson.train(args)
    with open(path) as f:
        records = [json.loads(line) for line in f]
    config = records[0]["hyperparameters"]
    assert config["dtype"] == "torch.float64" and config["device"] == "cpu"
    assert config["opt.schedule"] == "none" and config["opt.weight_decay"] == 0
    assert config["opt.momentum"] == 0.9 and config["opt.damping"] == 0.001
    assert config["opt.residual_scaling"] == "1/sqrt(N_block)"
    assert config["sampling"] == ("fixed" if fixed else "resample_each_step")
    assert config["n_pde_aux"] == config["n_bc_aux_per_edge"] == 0
    assert torch.equal(train_batches[0][0], train_batches[1][0]) == fixed
    assert not torch.equal(val_batches[0][0], train_batches[0][0])
    assert not torch.equal(val_batches[0][0], val_batches[1][0])
    training = [r["metrics"] for r in records if r["type"] == "train"]
    assert len(training) == 3 and all(math.isfinite(m["loss"]) for m in training)
    if line_search:
        assert all(m["lr"] in config["opt.ls_grid"] for m in training)
    else:
        assert all(m["lr"] == 0.05 for m in training)
    assert [r["step"] for r in records if r["type"] == "val"] == [2, 3]
    assert records[-1]["completed"] and math.isfinite(records[-1]["final_rel_l2"])


@pytest.mark.parametrize("optimizer", ["gnome", "soap", "adamw"])
def test_existing_optimizers_still_train_and_evaluate_final_step(tmp_path, monkeypatch, optimizer):
    monkeypatch.setattr(poisson, "pick_device", lambda: torch.device("cpu"))
    args = poisson.parse_args(config_overrides(poisson, [
        "--optimizer", optimizer, "--hidden", "3", "--depth", "2", "--steps", "2",
        "--n-pde", "5", "--n-bc-per-edge", "2", "--log-every", "0",
        "--runs-dir", str(tmp_path), "--quiet",
    ]))
    with open(poisson.train(args)) as f:
        records = [json.loads(line) for line in f]
    assert records[-1]["completed"] and math.isfinite(records[-1]["final_rel_l2"])


@pytest.mark.parametrize("option,value", [
    ("--engdw-damping", "0"), ("--engdw-damping", "nan"),
    ("--engdw-momentum", "1"), ("--engdw-momentum", "nan"),
    ("--engdw-chunk", "0"), ("--lr", "inf"),
    ("--diagnostics-every", "1"), ("--measure-rho-every", "1"),
])
def test_invalid_engdw_options(option, value):
    with pytest.raises(SystemExit):
        poisson.parse_args(config_overrides(poisson, ["--optimizer", "engdw", option, value]))


@pytest.mark.parametrize("optimizer", ["gnome", "soap", "adamw", "engdw"])
@pytest.mark.parametrize("fixed", [False, True])
def test_fixed_dataset_across_optimizers(tmp_path, monkeypatch, optimizer, fixed):
    monkeypatch.setattr(poisson, "pick_device", lambda: torch.device("cpu"))
    reference = poisson.poisson_reference
    monkeypatch.setattr(poisson, "poisson_reference", lambda: reference(9, 8))
    training, validation = [], []
    original_residual, original_step, original_terms = (
        poisson.stacked_residuals, ENGDW.step, poisson.term_losses)

    def capture(points):
        training.append(tuple(t.detach().clone() for t in points))

    def residual(model, batch):
        capture(batch)
        return original_residual(model, batch)

    def step(self, batch):
        capture(batch)
        return original_step(self, batch)

    def terms(model, batch):
        validation.append(tuple(t.detach().clone() for t in batch))
        return original_terms(model, batch)

    monkeypatch.setattr(poisson, "stacked_residuals", residual)
    monkeypatch.setattr(ENGDW, "step", step)
    monkeypatch.setattr(poisson, "term_losses", terms)
    args = poisson.parse_args(config_overrides(poisson, [
        "--optimizer", optimizer, "--hidden", "3", "--depth", "2", "--steps", "3",
        "--n-pde", "5", "--n-bc-per-edge", "2", "--aux-frac", "0.4",
        "--warmup-steps", "0", "--log-every", "1", "--runs-dir", str(tmp_path), "--quiet",
    ] + (["--fixed-dataset"] if fixed else [])))
    with open(poisson.train(args)) as f:
        records = [json.loads(line) for line in f]
    # Identify main/aux batches by size, independently of Gnome closure order.
    main = [b for b in training if len(b[0]) == 5]
    aux = [b for b in training if len(b[0]) == 2]
    assert len(main) == 3
    for a, b in zip(main, main[1:]):
        assert all(torch.equal(x, y) for x, y in zip(a, b)) == fixed
    if optimizer == "gnome":
        assert len(aux) == 3
        assert any(not all(torch.equal(x, y) for x, y in zip(aux[0], b)) for b in aux[1:])
        for subset, parent in zip(aux, main):
            sub_xy = torch.cat(subset[:2], 1)
            main_xy = torch.cat(parent[:2], 1)
            membership = (sub_xy[:, None] == main_xy[None]).all(-1).any(-1)
            assert membership.all() if fixed else not membership.any()
            if fixed:
                for edge in range(4):
                    point = torch.cat(subset[2:], 1)[edge:edge+1]
                    candidates = torch.cat(parent[2:], 1)[2*edge:2*(edge+1)]
                    assert (point == candidates).all(-1).any()
    else:
        assert not aux
    assert len(validation) == 3
    assert not torch.equal(validation[0][0], main[0][0])
    assert not torch.equal(validation[0][0], validation[1][0])
    config = records[0]["hyperparameters"]
    assert config["fixed_dataset"] == fixed
    assert config["sampling"] == ("fixed" if fixed else "resample_each_step")
    assert config["aux_sampling"] == (("subsample_fixed_main_each_step" if fixed else "resample_each_step") if optimizer == "gnome" else None)
    assert records[-1]["completed"] and math.isfinite(records[-1]["final_rel_l2"])


@pytest.mark.parametrize("n_pde,n_edge", [(1, 1), (4, 2), (9, 5)])
def test_aux_subsets_preserve_pairs_edges_and_main_data(n_pde, n_edge):
    main = poisson.sample_batch(9, 5, torch.device("cpu"))
    saved = tuple(t.clone() for t in main)
    aux = poisson.sample_aux_from_batch(main, n_pde, n_edge)
    groups = [(torch.cat(main[:2], 1), torch.cat(aux[:2], 1), n_pde)]
    boundary, subset = torch.cat(main[2:], 1), torch.cat(aux[2:], 1)
    groups += [(boundary[i*5:(i+1)*5], subset[i*n_edge:(i+1)*n_edge], n_edge) for i in range(4)]
    for parent, child, count in groups:
        assert child.shape == (count, 2)
        assert torch.unique(child, dim=0).shape[0] == count
        assert (child[:, None] == parent[None]).all(-1).any(-1).all()
        if len(parent) == count:
            assert (parent[:, None] == child[None]).all(-1).any(-1).all()
    for before, after in zip(saved, main):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("fraction", ["0", "-0.1", "1.1", "nan", "inf"])
def test_invalid_fixed_aux_fraction(fraction):
    with pytest.raises(SystemExit):
        poisson.parse_args(config_overrides(poisson, ["--optimizer", "gnome", "--fixed-dataset", "--aux-frac", fraction]))
