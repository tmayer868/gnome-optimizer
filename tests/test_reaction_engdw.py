from config_helpers import config_overrides

import json
import math

import pytest
import torch
from torch.func import jacrev
from torch.nn.utils import parameters_to_vector

from experiments.baselines import ENGDW
from experiments.pinns import reaction_pinn as reaction


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
    model = reaction.build_model(arch, reaction.build_embedding(embed), 3, 3)
    batch = reaction.sample_batch(7, 4, 3, torch.device("cpu"))
    fn, _ = reaction.make_engdw_functions(model, 5, embed != "periodic")
    return model, batch, fn


@pytest.mark.parametrize("arch", ["mlp", "modified", "fused", "fused-modified"])
@pytest.mark.parametrize("embed", ["none", "periodic"])
def test_functional_residual_and_jacobian_match_autograd(arch, embed):
    model, batch, fn = setup_problem(arch, embed)
    params = dict(model.named_parameters())
    actual = fn(params, batch).flatten()
    stacked = reaction.stacked_residuals(model, batch, 5, embed != "periodic")
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
    _, jacobian = reaction.make_engdw_functions(model, 5, embed != "periodic", 3)
    efficient = jacobian(params, batch)
    for key in params:
        torch.testing.assert_close(efficient[key], jd[key], atol=1e-11, rtol=1e-10)


def test_exact_logistic_solution_has_zero_functional_residual():
    class Exact(torch.nn.Module):
        def forward(self, t, x):
            return reaction.exact_solution(t, x, 5)

    model = Exact()
    batch = reaction.sample_batch(17, 5, 6, torch.device("cpu"))
    fn, _ = reaction.make_engdw_functions(model, 5)
    assert fn({}, batch).abs().max() < 1e-14


@pytest.mark.parametrize("chunk", [None, 1, 4])
@pytest.mark.parametrize("per_sample", [False, True])
@pytest.mark.parametrize("momentum", [0.0, 0.95])
def test_woodbury_step_matches_augmented_least_squares(chunk, per_sample, momentum):
    model, batch, fn = setup_problem()
    params = tuple(model.parameters())
    r = reaction.stacked_residuals(model, batch, 5)
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
    _, jacobian = reaction.make_engdw_functions(model, 5, chunk_size=chunk)
    opt = ENGDW(model, fn, lr=lr, damping=damping, chunk_size=chunk,
                jacobian_fn=jacobian if per_sample else None, momentum=momentum)
    opt.d_prev.copy_(previous_direction)
    before = r.detach().square().sum().item()
    loss = opt.step(batch)
    assert loss == pytest.approx(before)
    torch.testing.assert_close(parameters_to_vector(params), theta - lr * delta,
                               rtol=1e-10, atol=1e-11)


def test_line_search_uses_same_batch_and_best_grid_candidate():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    def residual(params, target):
        return (params["weight"].flatten() - target).unsqueeze(1)

    grid = [0.01, 0.1, 1.0]
    opt = ENGDW(model, residual, damping=0.1, lr=20,
                line_search=True, ls_grid=grid)
    target = torch.tensor([0.5])
    delta = (2 - 0.5) / 1.1
    expected_eta = min(grid, key=lambda e: (2 - e * delta - 0.5) ** 2)
    assert opt.step(target) == pytest.approx(1.5 ** 2)
    assert opt.param_groups[0]["lr"] == expected_eta
    assert model.weight.item() == pytest.approx(2 - expected_eta * delta)


@pytest.mark.parametrize("optimizer", ["engdw", "gnome", "soap", "adamw"])
def test_training_and_metadata(tmp_path, monkeypatch, optimizer):
    # ENGD-W must promote precision even when the caller did not request it.
    torch.set_default_dtype(torch.float32)
    monkeypatch.setattr(reaction, "pick_device", lambda: torch.device("cpu"))
    reference = reaction.reaction_reference
    monkeypatch.setattr(reaction, "reaction_reference",
                        lambda rho: reference(rho, nt=9, nx=8))
    args = reaction.parse_args(config_overrides(reaction, [
        "--optimizer", optimizer, "--steps", "3", "--hidden", "3", "--depth", "3",
        "--n-pde", "7", "--n-ic", "4", "--n-bc", "3", "--lr", "0.05",
        "--engdw-chunk", "3", "--log-every", "0", "--runs-dir", str(tmp_path),
        "--quiet",
    ] + (["--engdw-momentum", "0.95"] if optimizer == "engdw" else [])))
    path = reaction.train(args)
    with open(path) as file:
        records = [json.loads(line) for line in file]
    meta = records[0]["hyperparameters"]
    assert meta["rho"] == 5 and meta["sampling"] == "resample_each_step"
    if optimizer == "engdw":
        assert meta["dtype"] == "torch.float64"
        assert meta["opt.schedule"] == "none"
        assert meta["opt.weight_decay"] == 0
        assert meta["opt.momentum"] == 0.95
        assert meta["opt.residual_scaling"] == "1/sqrt(N_block)"
    train = [r["metrics"] for r in records if r["type"] == "train"]
    assert len(train) == 3
    assert all(math.isfinite(r["loss"]) for r in train)
    assert train[-1]["training_seconds"] >= train[0]["training_seconds"] > 0
    end = records[-1]
    assert end["completed"] and math.isfinite(end["final_rel_l2"])
    assert end["training_seconds"] <= end["wall_time_seconds"]


def test_poisson_uses_shared_solver_and_still_steps():
    from experiments.pinns import poisson5d_pinn as poisson

    assert poisson.ENGDW is ENGDW
    model = poisson.PINN(2, hidden=3, depth=2).double()
    fn = poisson.make_functional_residual(model, 2, False)
    opt = ENGDW(model, fn, damping=1e-4, chunk_size=2)
    interior = torch.rand(4, 2)
    boundary = torch.tensor([[0.0, 0.3], [1.0, 0.7]])
    before = parameters_to_vector(model.parameters()).detach().clone()
    assert math.isfinite(opt.step(interior, boundary))
    assert not torch.equal(parameters_to_vector(model.parameters()), before)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("momentum", [0.0, 0.95])
def test_engdw_training_points_and_validation_sampling(tmp_path, monkeypatch, fixed, momentum):
    monkeypatch.setattr(reaction, "pick_device", lambda: torch.device("cpu"))
    reference = reaction.reaction_reference
    monkeypatch.setattr(reaction, "reaction_reference",
                        lambda rho: reference(rho, nt=9, nx=8))
    train_batches, val_batches = [], []
    original_step, original_terms = ENGDW.step, reaction.term_losses

    def capture_step(self, batch):
        train_batches.append(tuple(t.clone() for t in batch))
        return original_step(self, batch)

    def capture_terms(model, batch, rho):
        val_batches.append(tuple(t.clone() for t in batch))
        return original_terms(model, batch, rho)

    monkeypatch.setattr(ENGDW, "step", capture_step)
    monkeypatch.setattr(reaction, "term_losses", capture_terms)
    args = reaction.parse_args(config_overrides(reaction, [
        "--optimizer", "engdw", "--steps", "3", "--hidden", "3", "--depth", "3",
        "--n-pde", "7", "--n-ic", "4", "--n-bc", "3", "--engdw-line-search",
        "--engdw-momentum", str(momentum),
        "--log-every", "1", "--runs-dir", str(tmp_path), "--quiet",
    ] + (["--engdw-fixed"] if fixed else [])))
    with open(reaction.train(args)) as file:
        records = [json.loads(line) for line in file]
    assert records[0]["hyperparameters"]["sampling"] == (
        "fixed" if fixed else "resample_each_step"
    )
    assert len(train_batches) == len(val_batches) == 3
    for batch in train_batches[1:]:
        assert all(torch.equal(a, b) == fixed for a, b in zip(batch, train_batches[0]))
    for i, batch in enumerate(val_batches):
        assert all(not torch.equal(a, b) for a, b in zip(batch, train_batches[i]))
        if i:
            assert all(not torch.equal(a, b) for a, b in zip(batch, val_batches[0]))
    assert records[-1]["completed"]


@pytest.mark.parametrize("momentum", [0.9, 0.95, 0.99])
@pytest.mark.parametrize("line_search", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_spring_directions_across_steps_and_reset(monkeypatch, momentum, line_search, fallback):
    model = torch.nn.Linear(3, 1, bias=False)

    def residual(params, inputs, target):
        return inputs @ params["weight"].T - target

    opt = ENGDW(model, residual, lr=0.17, damping=0.2, momentum=momentum,
                line_search=line_search, ls_grid=[0.1, 0.4, 0.8])
    assert torch.equal(opt.d_prev, torch.zeros(3))
    if fallback:
        def fail_cholesky(*args, **kwargs):
            raise torch.linalg.LinAlgError("exercise least-squares fallback")
        monkeypatch.setattr(torch.linalg, "cholesky", fail_cholesky)

    previous_direction = torch.zeros(3)
    for step in range(4):
        if step == 3:
            weights = model.weight.detach().clone()
            opt.reset_state()
            assert torch.equal(model.weight, weights)
            assert torch.equal(opt.d_prev, torch.zeros(3))
            previous_direction.zero_()
        # Changing sample count and Jacobian tests that history lives in
        # parameter space, while its center is projected through the current J.
        J = torch.randn(2 + step % 2, 3)
        target = torch.randn(J.shape[0], 1)
        theta = model.weight.detach().flatten().clone()
        r = (J @ theta - target.flatten())
        A = torch.cat([J, math.sqrt(opt.lam) * torch.eye(3)])
        b = torch.cat([r, math.sqrt(opt.lam) * momentum * previous_direction])
        direction = torch.linalg.lstsq(A, b, driver="gelsd").solution
        eta = min(((J @ (theta - e * direction) - target.flatten()).square().sum().item(), e)
                  for e in opt.ls_grid)[1] if line_search else opt.lr
        assert opt.step(J, target) == pytest.approx(r.square().sum().item())
        torch.testing.assert_close(model.weight.flatten(), theta - eta * direction,
                                   atol=1e-12, rtol=1e-11)
        torch.testing.assert_close(opt.d_prev, direction, atol=1e-12, rtol=1e-11)
        assert opt.param_groups[0]["lr"] == eta
        assert not opt.d_prev.requires_grad and opt.d_prev.grad_fn is None
        previous_direction = direction


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("line_search", [False, True])
def test_zero_momentum_matches_original_arithmetic_exactly(dtype, line_search):
    model = torch.nn.Linear(3, 1, bias=False).to(dtype)
    def residual(params, J, target):
        return J @ params["weight"].T - target
    opt = ENGDW(model, residual, lr=0.17, damping=0.2, line_search=line_search,
                ls_grid=[0.1, 0.4, 0.8])
    assert opt.momentum == 0.0
    for _ in range(3):
        inputs, target = torch.randn(2, 3, dtype=dtype), torch.randn(2, 1, dtype=dtype)
        params = {k: p.detach() for k, p in model.named_parameters()}
        r = residual(params, inputs, target).squeeze(-1)
        jd = jacrev(lambda p: residual(p, inputs, target).squeeze(-1))(params)
        J = torch.cat([jd[k].reshape(2, -1) for k in params], dim=1)
        # Original ENGD-W arithmetic, including its update and grid selection.
        G = J @ J.T
        G = 0.5 * (G + G.T)
        G.diagonal().add_(opt.lam)
        direction = J.T @ torch.cholesky_solve(r.unsqueeze(1), torch.linalg.cholesky(G)).squeeze(1)
        eta = min((opt._loss_at(e, direction, (inputs, target)), e)
                  for e in opt.ls_grid)[1] if line_search else opt.lr
        expected = model.weight.detach().clone() - (eta * direction).view_as(model.weight)
        # Disabled momentum must not depend on stale history, even non-finite.
        opt.d_prev.fill_(float("nan"))
        assert opt.step(inputs, target) == float(r.square().sum())
        assert torch.equal(model.weight, expected)
        assert torch.equal(opt.d_prev, direction)


@pytest.mark.parametrize("momentum", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_momentum(momentum):
    model, batch, fn = setup_problem()
    with pytest.raises(ValueError, match="momentum"):
        ENGDW(model, fn, momentum=momentum)
    with pytest.raises(SystemExit):
        args = reaction.parse_args(config_overrides(reaction, ["--optimizer", "engdw", "--engdw-momentum", str(momentum)]))
        reaction.train(args)


def test_momentum_cli_default():
    assert reaction.parse_args(config_overrides(reaction, ["--optimizer", "engdw"])).engdw_momentum == 0.0


def test_engdw_fixed_rejects_other_optimizers():
    args = reaction.parse_args(config_overrides(reaction, ["--optimizer", "gnome", "--engdw-fixed"]))
    with pytest.raises(SystemExit, match="--engdw-fixed requires --optimizer engdw"):
        reaction.train(args)
