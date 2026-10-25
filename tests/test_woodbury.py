from config_helpers import config_overrides

"""Small explicit parameter-space solves are test oracles, never optimizer state."""

from collections import deque
import io
import itertools
import json
import math
from pathlib import Path

import pytest
import torch

from gnome import Gnome
from gnome.experimental import WoodburyGGN
from experiments.pinns import poisson_pinn as poisson


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(threads)


def explicit_direction(history, x, beta2, damping):
    h = damping * torch.eye(x.numel(), dtype=x.dtype, device=x.device)
    # Independent normalized geometric weights, newest has exponent zero.
    w = torch.tensor([beta2 ** k for k in reversed(range(len(history)))], dtype=x.dtype)
    if len(history):
        w /= w.sum()
    for sample, weight in zip(history, w):
        s = sample.reshape(-1).to(x)
        h += weight * torch.outer(s, s)
    return torch.linalg.solve(h, x.reshape(-1)).reshape_as(x)


@pytest.mark.parametrize("beta1", [0.0, 0.8])
@pytest.mark.parametrize("beta2", [0.0, 0.6, 0.99])
def test_directions_gram_cache_and_matrix_state(beta1, beta2):
    p = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float64))
    opt = WoodburyGGN([p], trust_radius=None, curvature_warmup_steps=0, lr=0.02, beta1=beta1, beta2=beta2, damping=0.05, max_history=3)
    history, momentum = [], torch.zeros_like(p)
    for step in range(1, 11):
        g, sample = torch.randn_like(p), torch.randn_like(p, requires_grad=True) * 2
        history = (history + [sample.detach().clone()])[-3:]
        momentum = beta1 * momentum + (1 - beta1) * g
        x = momentum / (1 - beta1 ** step)
        before = p.detach().clone()
        p.grad = g
        opt.step([sample])
        expected = explicit_direction(history, x, beta2, 0.05)
        torch.testing.assert_close((before - p) / 0.02, expected, rtol=2e-11, atol=2e-11)
        state = opt.state[p]
        assert isinstance(state["history"], deque)
        assert state["step"] == step
        assert len(state["history"]) == min(step, 3)
        for stored, original in zip(state["history"], history):
            assert stored.shape == p.shape and stored.grad_fn is None and not stored.requires_grad
            torch.testing.assert_close(stored, original)
        gram = torch.stack([torch.stack([(a * b).sum() for b in history]) for a in history])
        torch.testing.assert_close(state["gram"], gram, rtol=1e-14, atol=1e-14)
        # State must own the raw samples even if a caller reuses its buffer.
        with torch.no_grad():
            sample.zero_()
        torch.testing.assert_close(state["history"][-1], history[-1])


@pytest.mark.parametrize("beta2", [0.0, 0.5, 0.99, 0.9999999])
def test_history_size_and_normalized_ema(beta2):
    p = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    opt = WoodburyGGN([p], trust_radius=None, curvature_warmup_steps=0, beta2=beta2)
    m = opt.param_groups[0]["history_size"]
    assert m == (math.ceil(math.log(0.01) / math.log(beta2)) if beta2 else 1)
    for n in {1, min(m, 7), min(m, 459)}:
        weights = opt._weights(n, beta2, p)
        assert weights.sum().item() == pytest.approx(1, abs=2e-15)
        assert (weights[0] / weights[-1]).item() == pytest.approx(beta2 ** (n - 1))
    if beta2:
        assert beta2 ** m <= 0.01
        assert beta2 ** (m - 1) > 0.01


def test_empty_history_and_sherman_morrison():
    p = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float64))
    damping = 0.2
    opt = WoodburyGGN([p], trust_radius=None, curvature_warmup_steps=0, lr=1, damping=damping)
    x, s = torch.randn_like(p), torch.randn_like(p)
    p.grad = x
    opt.step()
    torch.testing.assert_close(p, -x / damping)
    assert len(opt.state[p]["history"]) == 0
    before = p.detach().clone()
    opt.step(surrogate_grads=[s])
    expected = (x - s * (s * x).sum() / (damping + s.square().sum())) / damping
    torch.testing.assert_close(before - p, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_checkpoint_preserves_gram_precision_and_continues(dtype):
    p = torch.nn.Parameter(torch.zeros(2, 3, dtype=dtype))
    opt = WoodburyGGN([p], trust_radius=None, curvature_warmup_steps=0, lr=0.001, beta1=0.8, max_history=2, gram_dtype=torch.float64)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step([torch.randn_like(p)])
    stream = io.BytesIO()
    torch.save(opt.state_dict(), stream)
    stream.seek(0)
    q = torch.nn.Parameter(p.detach().clone())
    restored = WoodburyGGN([q])
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert restored._step_count == 3
    assert restored.state[q]["gram"].dtype == torch.float64
    assert restored.state[q]["history"][0].dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    torch.testing.assert_close(restored.state[q]["gram"], opt.state[p]["gram"], rtol=0, atol=0)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        sample = torch.randn_like(p)
        opt.step([sample])
        restored.step([sample])
        torch.testing.assert_close(q, p, rtol=0, atol=0)
        torch.testing.assert_close(restored.state[q]["gram"], opt.state[p]["gram"], rtol=0, atol=0)


def test_surrogate_covariance_equals_explicit_mlp_ggn(monkeypatch):
    """Enumerate every probe: no Monte Carlo tolerance hides a scale error."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 3, bias=False), torch.nn.Tanh(),
                                torch.nn.Linear(3, 1, bias=False)).double()
    params = list(model.parameters())
    x = torch.tensor([[0.2, -0.5], [0.8, 0.3], [-0.3, 0.7]], dtype=torch.float64)
    y = model(x)
    rows = [torch.autograd.grad(row, params, retain_graph=True) for row in y[:, 0]]
    jacobians = [torch.stack([row[i].flatten() for row in rows]) for i in range(len(params))]
    covariances = [torch.zeros((p.numel(), p.numel()), dtype=p.dtype) for p in params]
    opt = WoodburyGGN(params)
    assert WoodburyGGN._build_surrogate is Gnome._build_surrogate
    assert WoodburyGGN._compute_loss is Gnome._compute_loss
    for signs in itertools.product((-1.0, 1.0), repeat=len(x)):
        probe = torch.tensor(signs, dtype=x.dtype)[:, None]
        monkeypatch.setattr("gnome.optimizer._rademacher_like", lambda like: probe.to(like))
        surrogate = opt._build_surrogate(model(x), torch.arange(len(x)))
        samples = torch.autograd.grad(surrogate, params)
        for cov, sample in zip(covariances, samples):
            cov.add_(torch.outer(sample.flatten(), sample.flatten()) / (2 ** len(x)))
    for cov, jacobian in zip(covariances, jacobians):
        torch.testing.assert_close(cov, 2 * jacobian.T @ jacobian / len(x), rtol=1e-13, atol=1e-14)


def test_closures_keep_gradients_separate_and_match_explicit_blocks():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3, bias=False), torch.nn.Tanh(),
                                torch.nn.Linear(3, 1, bias=False)).double()
    params = list(model.parameters())
    x, target = torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 1, dtype=torch.float64)
    opt = WoodburyGGN(params, trust_radius=None, curvature_warmup_steps=0, lr=0.01, damping=0.2)
    loss = (model(x) - target).square().mean()
    ordinary = torch.autograd.grad(loss, params)
    rng = torch.get_rng_state()
    sample = torch.autograd.grad(opt._build_surrogate(model(x[:3]), torch.arange(3)), params)
    torch.set_rng_state(rng)
    before = [p.detach().clone() for p in params]
    # Existing .grad buffers must neither enter the step nor be contaminated.
    for p in params:
        p.grad = torch.full_like(p, 123)
    result = opt.step(lambda: (model(x), target), lambda: (model(x[:3]), target[:3]))
    torch.testing.assert_close(result, loss.detach())
    for p, old, g, s in zip(params, before, ordinary, sample):
        torch.testing.assert_close(old - p, 0.01 * explicit_direction([s], g, 0.99, 0.2),
                                   rtol=1e-11, atol=1e-12)
        assert (p.grad == 123).all()
        torch.testing.assert_close(opt.state[p]["history"][0], s)


def test_small_deterministic_mlp_training_matches_empirical_inverse_each_step():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3, bias=False), torch.nn.Tanh(),
                                torch.nn.Linear(3, 1, bias=False)).double()
    params = list(model.parameters())
    x = torch.tensor([[-0.8, 0.2], [0.4, -0.5], [0.8, 0.6]], dtype=torch.float64)
    target = torch.sin(x[:, :1]) + 0.2 * x[:, 1:]
    opt = WoodburyGGN(params, trust_radius=None, curvature_warmup_steps=0, lr=0.01, damping=0.1, beta2=0.8, max_history=10)
    start = (model(x) - target).square().mean().item()
    for _ in range(200):
        old = [p.detach().clone() for p in params]
        ordinary = torch.autograd.grad((model(x) - target).square().mean(), params)
        loss = opt.step(lambda: (model(x), target), lambda: (model(x), target))
        assert torch.isfinite(loss)
        for p, previous, g in zip(params, old, ordinary):
            exact = explicit_direction(list(opt.state[p]["history"]), g, 0.8, 0.1)
            torch.testing.assert_close((previous - p) / 0.01, exact, rtol=1e-9, atol=1e-12)
            assert opt.state[p]["solve_failures"] == opt.state[p]["jitter_events"] == 0
    final = (model(x) - target).square().mean().item()
    assert final < start * 0.05, (start, final)


def test_diagnostics_and_generic_solve_fallback(monkeypatch):
    p = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float64))
    records = []
    opt = WoodburyGGN([p], trust_radius=None, curvature_warmup_steps=0, lr=0.1, damping=0.2, diagnostics=records.append, diagnostics_every=2)
    original = torch.linalg.cholesky_ex
    def fail(matrix):
        factor, info = original(matrix)
        return factor, torch.ones_like(info)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", fail)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        old = p.detach().clone()
        opt.step([torch.randn_like(p)])
        exact = explicit_direction(list(opt.state[p]["history"]), p.grad, 0.99, 0.2)
        torch.testing.assert_close(old - p, 0.1 * exact)
    assert len(records) == 1
    rec = records[0]
    assert rec["step"] == 2 and rec["history_length"] == 2 and rec["history_max"] == 459
    assert rec["solve_failures"] == 2 and rec["jitter_events"] == 0
    assert rec["damping"] == 0.2 and rec["grad_norm"] == rec["x_norm"]
    assert rec["direction_ratio"] == pytest.approx(rec["direction_norm"] / rec["x_norm"])
    assert 0 < rec["cosine"] <= 1
    assert rec["condition_number"] == pytest.approx((rec["k_eig_max"] + 0.2) / (rec["k_eig_min"] + 0.2))
    json.dumps(rec, allow_nan=False)


def test_jitter_is_separate_from_damping_and_counted(monkeypatch):
    matrix = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    state = dict(solve_failures=0, jitter_events=0)
    damping = 0.1
    p = torch.nn.Parameter(torch.zeros(2, 2))
    opt = WoodburyGGN([p], trust_radius=None, curvature_warmup_steps=0, damping=damping)
    result = opt._solve(matrix, torch.ones(2, dtype=matrix.dtype), state)
    assert torch.isfinite(result).all()
    assert state == dict(solve_failures=1, jitter_events=1)
    assert opt.param_groups[0]["damping"] == damping


@pytest.mark.parametrize("gram_dtype", [None, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("trust_radius", [None, 0.1])
def test_float32_against_double_oracle(gram_dtype, device, trust_radius):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = torch.nn.Parameter(torch.zeros(4, 3, device=device, dtype=torch.float32))
    records = []
    opt = WoodburyGGN([p], trust_radius=trust_radius, curvature_warmup_steps=0, lr=1, damping=1e-3, gram_dtype=gram_dtype,
                      diagnostics=records.append, diagnostics_every=1)
    history = []
    for _ in range(4):
        p.grad = torch.randn_like(p)
        sample = torch.randn_like(p)
        history.append(sample.cpu().double())
        old = p.detach().clone()
        opt.step([sample])
        if trust_radius is None:
            exact = explicit_direction(history, p.grad.cpu().double(), 0.99, 1e-3)
        else:
            budget = trust_radius * (old.cpu().double().square().sum() + 1).sqrt().item()
            exact, _ = explicit_trust_direction(history, p.grad.cpu().double(), 0.99, 1e-3, budget)
            assert (old - p).norm().item() <= budget * (1 + 1e-5)
        actual = (old - p).cpu().double()
        torch.testing.assert_close(actual, exact, rtol=2e-4, atol=5e-4)
    assert opt.state[p]["history"][0].dtype == torch.float32
    assert len(records) == 4


def test_parameter_groups_have_independent_histories_and_damping():
    p = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float64))
    q = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    opt = WoodburyGGN([dict(params=[p], damping=0.1, max_history=1),
                       dict(params=[q], damping=0.5, max_history=3)], lr=0.01, curvature_warmup_steps=0, trust_radius=None)
    for _ in range(5):
        old = [t.detach().clone() for t in (p, q)]
        p.grad, q.grad = torch.randn_like(p), torch.randn_like(q)
        opt.step([torch.randn_like(p), torch.randn_like(q)])
        for group, previous, param in zip(opt.param_groups, old, (p, q)):
            exact = explicit_direction(list(opt.state[param]["history"]), param.grad,
                                       group["beta2"], group["damping"])
            torch.testing.assert_close(previous - param, 0.01 * exact)
    assert len(opt.state[p]["history"]) == 1
    assert len(opt.state[q]["history"]) == 3


@pytest.mark.parametrize("kwargs", [
    {"beta1": -1}, {"beta2": 1}, {"beta2": float("nan")}, {"history_tol": 0},
    {"history_tol": 1}, {"damping": 0}, {"damping": float("inf")},
    {"curvature_warmup_steps": -1}, {"curvature_warmup_steps": 1.5},
    {"curvature_warmup_steps": True}, {"lr": -1}, {"max_history": 0}, {"max_history": 1.5}, {"gram_dtype": torch.float16},
    {"trust_radius": 0}, {"trust_radius": -1}, {"trust_radius": float("nan")},
    {"trust_radius": float("inf")},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        WoodburyGGN([torch.nn.Parameter(torch.zeros(2, 3))], **kwargs)


@pytest.mark.parametrize("shape", [(2,), (2, 3, 4), ()])
def test_unsupported_parameters_rejected(shape):
    with pytest.raises(ValueError, match="2D"):
        WoodburyGGN([torch.nn.Parameter(torch.zeros(shape))])


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("arch", ["mlp", "fused"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_poisson_integration(tmp_path, monkeypatch, fixed, arch, dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    monkeypatch.setattr(poisson, "pick_device", lambda: torch.device("cpu"))
    reference = poisson.poisson_reference
    monkeypatch.setattr(poisson, "poisson_reference", lambda: reference(9, 8))
    batches = []
    residual = poisson.stacked_residuals
    def capture(model, batch):
        batches.append(tuple(t.detach().clone() for t in batch))
        return residual(model, batch)
    monkeypatch.setattr(poisson, "stacked_residuals", capture)
    args = poisson.parse_args(config_overrides(poisson, [
        "--optimizer", "woodbury", "--hidden", "3", "--depth", "4", "--arch", arch,
        "--steps", "4", "--n-pde", "8", "--n-bc-per-edge", "3", "--aux-frac", "0.5",
        "--woodbury-damping", "0.1", "--max-history", "2", "--gram-float64",
        "--woodbury-warmup-steps", "2",
        "--diagnostics-every", "2", "--diagnostics-params", "0", "--warmup-steps", "0",
        "--log-every", "2", "--runs-dir", str(tmp_path), "--quiet",
    ] + (["--fixed-dataset"] if fixed else [])))
    try:
        path = poisson.train(args)
    finally:
        torch.set_default_dtype(old_dtype)
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    cfg = records[0]["hyperparameters"]
    assert cfg["opt.beta1"] == 0 and cfg["opt.history_size"] == 2
    assert cfg["opt.damping"] == 0.1 and cfg["opt.weight_decay"] == 0
    assert cfg["opt.trust_radius"] == 1.0
    assert cfg["opt.curvature_warmup_steps"] == 2
    training_lrs = [r["metrics"]["lr"] for r in records if r["type"] == "train"]
    assert training_lrs[:2] == [0, 0]
    assert all(lr > 0 for lr in training_lrs[2:])
    validation_lrs = [r["metrics"]["lr"] for r in records if r["type"] == "val"]
    assert validation_lrs == [training_lrs[1], training_lrs[3]]
    assert cfg["opt.gram_dtype"] == "float64"
    assert cfg["aux_sampling"] == ("subsample_fixed_main_each_step" if fixed else "resample_each_step")
    assert records[-1]["completed"] and math.isfinite(records[-1]["final_rel_l2"])
    assert len(batches) == 8  # main and aux each step
    main, aux = batches[::2], batches[1::2]
    assert all(torch.equal(a, b) for a, b in zip(main[0], main[1])) == fixed
    assert any(not torch.equal(aux[0][0], batch[0]) for batch in aux[1:])
    for subset, parent in zip(aux, main):
        s, p = torch.cat(subset[:2], 1), torch.cat(parent[:2], 1)
        membership = (s[:, None] == p[None]).all(-1).any(-1)
        assert membership.all() if fixed else not membership.any()
        if fixed:
            s, p = torch.cat(subset[2:], 1), torch.cat(parent[2:], 1)
            for edge in range(4):
                assert (s[edge:edge+1, None] == p[None, edge*3:(edge+1)*3]).all(-1).any()
    diag_path, = Path(path).parent.glob("*.diag.jsonl")
    diag = [json.loads(line) for line in diag_path.read_text().splitlines()]
    assert [r["step"] for r in diag] == [2, 4]
    assert all(r["param"] == 0 and r["history_length"] == 2 for r in diag)
    assert [r["curvature_warmup"] for r in diag] == [True, False]
    assert [r["lr"] for r in diag] == validation_lrs
    assert all(r["trust_ratio"] <= 1 + 1e-5 and r["damping"] >= 0.1 for r in diag)


@pytest.mark.parametrize("option,value", [
    ("--woodbury-warmup-steps", "-1"),
    ("--trust-region", "-1"), ("--trust-region", "nan"), ("--trust-region", "inf"),
    ("--woodbury-damping", "0"), ("--woodbury-damping", "nan"), ("--history-tol", "1"),
    ("--max-history", "0"), ("--beta1", "nan"), ("--beta2", "1"), ("--lr", "inf"),
    ("--aux-frac", "0"), ("--aux-frac", "1.1"), ("--measure-rho-every", "1"),
])
def test_poisson_invalid_woodbury_options(option, value):
    with pytest.raises(SystemExit):
        poisson.parse_args(config_overrides(poisson, ["--optimizer", "woodbury", "--fixed-dataset", option, value]))


def test_default_warmup_freezes_exactly_100_steps_and_collects_state(monkeypatch):
    p = torch.nn.Parameter(torch.randn(2, 3, dtype=torch.float64))
    original = p.detach().clone()
    opt = WoodburyGGN([p], trust_radius=None, lr=0.01, beta1=0.8, max_history=3, damping=0.2)
    precondition = opt._precondition
    def unexpected_solve(*args):
        pytest.fail("Warmup without diagnostics should not solve")
    monkeypatch.setattr(opt, "_precondition", unexpected_solve)
    momentum = torch.zeros_like(p)
    for step in range(1, 102):
        if step == 101:
            monkeypatch.setattr(opt, "_precondition", precondition)
        p.grad = torch.randn_like(p)
        momentum = 0.8 * momentum + 0.2 * p.grad
        opt.step([torch.randn_like(p)])
        state = opt.state[p]
        assert state["step"] == step
        assert len(state["history"]) == min(step, 3)
        torch.testing.assert_close(state["momentum"], momentum)
        assert opt.param_groups[0]["lr"] == 0.01
        if step <= 100:
            torch.testing.assert_close(p, original, rtol=0, atol=0)
            assert opt.param_groups[0]["applied_lr"] == 0
        else:
            exact = explicit_direction(list(state["history"]), momentum / (1 - 0.8 ** step), 0.99, 0.2)
            torch.testing.assert_close(p, original - 0.01 * exact)
            assert opt.param_groups[0]["applied_lr"] == 0.01


@pytest.mark.parametrize("checkpoint_step", [1, 3, 4])
def test_warmup_resume_and_external_lr_schedule(checkpoint_step):
    p = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float64))
    opt = WoodburyGGN([p], trust_radius=None, lr=0.01, curvature_warmup_steps=3, max_history=2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    for _ in range(checkpoint_step):
        p.grad = torch.ones_like(p)
        opt.step([torch.randn_like(p)])
        scheduler.step()
    q = torch.nn.Parameter(p.detach().clone())
    other = WoodburyGGN([q])
    other_scheduler = torch.optim.lr_scheduler.StepLR(other, step_size=2, gamma=0.5)
    stream = io.BytesIO()
    torch.save(opt.state_dict(), stream)
    stream.seek(0)
    other.load_state_dict(torch.load(stream, weights_only=True))
    other_scheduler.load_state_dict(scheduler.state_dict())
    for step in range(checkpoint_step + 1, 7):
        before = p.detach().clone()
        expected_lr = opt.param_groups[0]["lr"] if step > 3 else 0
        p.grad = q.grad = torch.ones_like(p)
        sample = torch.randn_like(p)
        opt.step([sample])
        other.step([sample])
        assert opt.param_groups[0]["applied_lr"] == other.param_groups[0]["applied_lr"] == expected_lr
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert torch.equal(p, before) == (step <= 3)
        scheduler.step()
        other_scheduler.step()


def explicit_trust_direction(history, x, beta2, floor, budget):
    """Independent parameter-space bisection, only used as a test oracle."""
    candidate = explicit_direction(history, x, beta2, floor)
    if candidate.norm() <= budget:
        return candidate, floor
    lo, hi = floor, max(floor, x.norm().item() / budget)
    for _ in range(70):
        mid = (lo + hi) / 2
        if explicit_direction(history, x, beta2, mid).norm() > budget:
            lo = mid
        else:
            hi = mid
    return explicit_direction(history, x, beta2, hi), hi


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gram_dtype", [None, torch.float64])
@pytest.mark.parametrize("n", [0, 1, 5, 20])
@pytest.mark.parametrize("radius", [0.02, 5.0, 1e6])
def test_trust_region_matches_dense_oracle(dtype, gram_dtype, n, radius):
    p = torch.nn.Parameter(torch.randn(4, 3, dtype=dtype))
    start = p.detach().clone()
    records = []
    opt = WoodburyGGN([p], lr=0.3, curvature_warmup_steps=0, trust_radius=radius,
                      damping=1e-3, max_history=20,
                      gram_dtype=gram_dtype,
                      diagnostics=records.append, diagnostics_every=1)
    samples = [torch.randn_like(p) for _ in range(n)]
    for sample in samples:
        opt.step([sample])  # history only: no ordinary gradient yet
    p.grad = torch.randn_like(p)
    budget = radius * (start.double().square().sum() + 1).sqrt().item()
    expected, damping = explicit_trust_direction(
        [s.double() for s in samples], p.grad.double(), 0.99, 1e-3, budget)
    opt.step()
    actual = (start - p).double() / 0.3
    tolerance = 3e-4 if dtype == torch.float32 and gram_dtype is None else 4e-6
    # Frobenius-relative agreement: float32 Gram/reconstruction error at the
    # floor is amplified by 1/damping, especially for small individual entries.
    assert (actual - expected).norm() <= tolerance * expected.norm() + 1e-7
    assert actual.norm() <= budget * (1 + 1e-5)
    assert opt.state[p]["lm_lambda"] == pytest.approx(damping, rel=tolerance)
    rec, = records
    assert rec["trust_budget"] == pytest.approx(budget, rel=1e-6)
    assert rec["trust_binding"] == (damping > 1e-3)
    assert rec["trust_ratio"] <= 1 + 1e-5
    assert rec["damping_floor"] == opt.param_groups[0]["damping"] == 1e-3
    assert rec["condition_number"] == pytest.approx(
        (rec["k_eig_max"] + rec["damping"]) / (rec["k_eig_min"] + rec["damping"]))
    if damping > 1e-3:
        assert rec["trust_ratio"] == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize("gradient_kind", ["zero", "span", "orthogonal"])
def test_trust_rank_deficient_and_cancellation_cases(gradient_kind):
    p = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    s = torch.tensor([[1., 2.], [0., 0.]], dtype=p.dtype)
    opt = WoodburyGGN([p], lr=1, damping=1e-10, trust_radius=0.1,
                      curvature_warmup_steps=0, max_history=3)
    for _ in range(3):
        opt.step([s])
    p.grad = {"zero": torch.zeros_like(p), "span": s,
              "orthogonal": torch.tensor([[0., 0.], [1., -2.]], dtype=p.dtype)}[gradient_kind]
    expected, lam = explicit_trust_direction([s] * 3, p.grad, 0.99, 1e-10, 0.1)
    opt.step()
    torch.testing.assert_close(-p, expected, rtol=1e-6, atol=1e-9)
    assert p.norm() <= 0.1 * (1 + 1e-6)
    assert opt.state[p]["lm_lambda"] == pytest.approx(lam, rel=2e-6)


def test_trust_corrects_underestimated_history_norm():
    """The final matrix-norm guard must recover from an inaccurate cached Gram."""
    p = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    opt = WoodburyGGN([p], lr=1, damping=0.001, trust_radius=0.1, curvature_warmup_steps=0)
    s = torch.ones_like(p)
    opt.step([s])
    # Simulate an inner-product cache error large enough to affect the norm.
    opt.state[p]["gram"].mul_(0.95)
    p.grad = s.clone()
    opt.step()
    assert p.norm() <= 0.1 * (1 + 1e-6)
    assert opt.state[p].get("trust_fallbacks", 0) == 1


def test_adaptive_damping_resume_momentum_and_independent_groups():
    p = torch.nn.Parameter(torch.randn(2, 3, dtype=torch.float64))
    q = torch.nn.Parameter(torch.randn(3, 2, dtype=torch.float64))
    opt = WoodburyGGN([dict(params=[p], trust_radius=0.02), dict(params=[q], trust_radius=10)],
                      lr=0.1, beta1=0.8, curvature_warmup_steps=2, max_history=3)
    for _ in range(4):
        p.grad, q.grad = torch.randn_like(p), torch.randn_like(q)
        opt.step([torch.randn_like(p), torch.randn_like(q)])
    replacements = [torch.nn.Parameter(v.detach().clone()) for v in (p, q)]
    restored = WoodburyGGN([dict(params=[r]) for r in replacements])
    stream = io.BytesIO()
    torch.save(opt.state_dict(), stream)
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    for _ in range(3):
        samples = [torch.randn_like(v) for v in (p, q)]
        for v, r in zip((p, q), replacements):
            v.grad = torch.randn_like(v)
            r.grad = v.grad.clone()
        before = [v.detach().clone() for v in (p, q)]
        opt.step(samples)
        restored.step(samples)
        for v, r, old, group in zip((p, q), replacements, before, opt.param_groups):
            torch.testing.assert_close(v, r, rtol=0, atol=0)
            state = opt.state[v]
            x = state["momentum"] / (1 - 0.8 ** state["step"])
            expected, lam = explicit_trust_direction(list(state["history"]), x, 0.99, 1e-3,
                                                     group["trust_radius"] * (old.square().sum() + 1).sqrt())
            torch.testing.assert_close((old - v) / 0.1, expected, rtol=5e-6, atol=1e-8)
            assert state["lm_lambda"] == restored.state[r]["lm_lambda"]
            assert state["lm_lambda"] == pytest.approx(lam, rel=5e-6)


def test_poisson_trust_disabled_restores_fixed_damping():
    args = poisson.parse_args(config_overrides(poisson, ["--optimizer", "woodbury", "--trust-region", "0"]))
    opt, cfg, _ = poisson.build_optimizer(
        "woodbury", [torch.nn.Parameter(torch.zeros(2, 2))], args.lr, args.weight_decay,
        warmup=0, total_steps=10, cosine_decay=1, trust_region=args.trust_region)
    assert opt.param_groups[0]["trust_radius"] is cfg["trust_radius"] is None


def test_legacy_checkpoint_does_not_enable_trust_region():
    p = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    opt = WoodburyGGN([p], trust_radius=None, curvature_warmup_steps=0)
    p.grad = torch.ones_like(p)
    opt.step()
    checkpoint = opt.state_dict()
    del checkpoint["param_groups"][0]["trust_radius"]
    q = torch.nn.Parameter(p.detach().clone())
    restored = WoodburyGGN([q])
    restored.load_state_dict(checkpoint)
    assert restored.param_groups[0]["trust_radius"] is None
    q.grad = p.grad.clone()
    opt.step()
    restored.step()
    torch.testing.assert_close(q, p, rtol=0, atol=0)
