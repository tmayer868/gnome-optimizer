from config_helpers import config_overrides

import copy
import json
import math

import pytest
import torch

from experiments.pinns import allen_cahn_pinn as ac


@pytest.fixture(autouse=True)
def restore_torch_settings():
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def assert_preserved(before, after):
    if isinstance(before, torch.Tensor):
        torch.testing.assert_close(after.cpu(), before.cpu().to(after.dtype),
                                   rtol=0, atol=0)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            assert_preserved(before[key], after[key])
    elif isinstance(before, (list, tuple)):
        assert len(before) == len(after)
        for a, b in zip(before, after):
            assert_preserved(a, b)
    else:
        assert before == after


@pytest.mark.parametrize("optimizer", ["gnome", "soap", "adamw"])
@pytest.mark.parametrize("arch,embed", [
    ("mlp", "none"), ("modified", "polynomial"),
    ("fused", "periodic"), ("fused-modified", "fourier"),
])
def test_training_switch_preserves_state(tmp_path, monkeypatch, optimizer, arch, embed):
    monkeypatch.setattr(ac, "pick_device", lambda: torch.device("cpu"))
    # Values deliberately not representable in FP32: the reference must retain
    # its original precision until evaluation in the FP64 phase.
    t_ref = torch.tensor([0., .123456789012345, 1.], dtype=torch.float64)
    x_ref = torch.tensor([-1., .123456789012345, 1.], dtype=torch.float64)
    monkeypatch.setattr(ac, "allen_cahn_reference", lambda **kw:
                        (t_ref, x_ref, torch.ones(3, 3, dtype=torch.float64)))
    build_optimizer, switch, evaluate = ac.build_optimizer, ac.switch_to_float64, ac.eval_rel_l2
    optimizers, schedulers, step_dtypes, step_counts, eval_dtypes = [], [], [], [], []
    switches = []

    def capture_optimizer(*args, **kwargs):
        opt, cfg, scheduler = build_optimizer(*args, **kwargs)
        optimizers.append(opt)
        schedulers.append(scheduler)
        original_step = opt.step

        def step(*args, **kwargs):
            step_dtypes.append(opt.param_groups[0]["params"][0].dtype)
            result = original_step(*args, **kwargs)
            step_counts.append(int(next(iter(opt.state.values()))["step"]))
            return result

        opt.step = step
        opt.step._wrapped_by_lr_sched = True
        return opt, cfg, scheduler

    def capture_switch(model, opt):
        before = copy.deepcopy(opt.state_dict())
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        schedule = copy.deepcopy(schedulers[0].state_dict())
        device = switch(model, opt)
        assert_preserved(before, opt.state_dict())
        assert_preserved(weights, model.state_dict())
        assert schedulers[0].state_dict() == schedule
        assert set(model.parameters()) == set(opt.state)
        for state in opt.state.values():
            for key in ("grad_m", "gnd_m", "exp_avg", "exp_avg_sq"):
                if key in state:
                    assert state[key].dtype == torch.float64
            for key in ("GG", "Q"):
                for tensor in state.get(key, []):
                    if isinstance(tensor, torch.Tensor):
                        assert tensor.dtype == torch.float64
        switches.append(device)
        return device

    def capture_eval(model, t, x, u, device):
        torch.testing.assert_close(t, t_ref, rtol=0, atol=0)
        torch.testing.assert_close(x, x_ref, rtol=0, atol=0)
        eval_dtypes.append(next(model.parameters()).dtype)
        return evaluate(model, t, x, u, device)

    monkeypatch.setattr(ac, "build_optimizer", capture_optimizer)
    monkeypatch.setattr(ac, "switch_to_float64", capture_switch)
    monkeypatch.setattr(ac, "eval_rel_l2", capture_eval)
    args = ac.parse_args(config_overrides(ac, [
        "--optimizer", optimizer, "--arch", arch, "--embed", embed,
        "--embed-dim", "4", "--embed-scale", "1", "--hidden", "4", "--depth", "3",
        "--float64-after", "2", "--steps", "23", "--warmup-steps", "4",
        "--n-pde", "5", "--n-ic", "4", "--n-bc", "3", "--log-every", "1",
        "--runs-dir", str(tmp_path), "--quiet",
    ]))
    # train() must start in FP32 even when the caller has another default,
    # and restore that default when done.
    torch.set_default_dtype(torch.float64)
    path = ac.train(args)
    assert torch.get_default_dtype() == torch.float64
    assert len(optimizers) == len(schedulers) == len(switches) == 1
    assert step_dtypes == eval_dtypes == [torch.float32] * 2 + [torch.float64] * 21
    assert all(b == a + 1 for a, b in zip(step_counts, step_counts[1:]))
    assert schedulers[0].last_epoch == 23
    # This crosses the optimizers' 20-step preconditioner refresh.
    assert all(torch.isfinite(p).all() for p in optimizers[0].state)
    with open(path) as stream:
        records = [json.loads(line) for line in stream]
    transitions = [r for r in records if r["type"] == "precision"]
    assert len(transitions) == 1 and transitions[0]["step"] == 2
    assert transitions[0]["metrics"]["dtype"] == "torch.float64"
    assert transitions[0]["metrics"]["lr"] > 0
    assert records[0]["hyperparameters"]["float64_after"] == 2
    assert records[-1]["completed"] and records[-1]["final_dtype"] == "torch.float64"
    assert math.isfinite(records[-1]["final_rel_l2"])


@pytest.mark.parametrize("flags", [
    ["--float64-after", "-1"], ["--float64-after", "3", "--steps", "3"],
    ["--float64-after", "4", "--steps", "3"],
    ["--float64", "--float64-after", "0"],
])
def test_invalid_precision_options(flags):
    with pytest.raises(SystemExit):
        ac.parse_args(config_overrides(ac, ["--optimizer", "gnome", *flags]))


@pytest.mark.parametrize("flags,dtype", [
    ([], torch.float32), (["--float64"], torch.float64),
    (["--float64-after", "0"], torch.float64),
])
def test_starting_precision_and_restoration(monkeypatch, flags, dtype):
    args = ac.parse_args(config_overrides(ac, ["--optimizer", "gnome", *flags]))
    def training(args):
        assert torch.get_default_dtype() == dtype
        raise RuntimeError("test cleanup")
    monkeypatch.setattr(ac, "_train", training)
    with pytest.raises(RuntimeError, match="test cleanup"):
        ac.train(args)
    assert torch.get_default_dtype() == torch.float32
