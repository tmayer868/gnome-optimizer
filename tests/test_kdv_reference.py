import sys

import numpy as np
import pytest
from scipy.io import savemat
import torch
import torch.nn as nn
import experiments.pinns.kdv_pinn as kdv_module

from experiments.pinns.kdv_pinn import (
    DEFAULT_MU_SQ,
    kdv_reference,
    load_kdv_reference,
    parse_args,
    pde_residual,
)


class _PolynomialSolution(nn.Module):
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return t * x**3


def test_pde_residual_uses_requested_mu_sq():
    model = _PolynomialSolution()
    t = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    x = torch.tensor([[-0.4], [0.3]], dtype=torch.float64)
    mu_a, mu_b = 1e-4, 9e-4

    residual_a = pde_residual(model, t, x, mu_a)
    residual_b = pde_residual(model, t, x, mu_b)

    # For u=t*x^3, u_xxx=6t, so changing only mu_sq has this exact effect.
    assert torch.allclose(
        residual_b - residual_a,
        (mu_b - mu_a) * 6.0 * t,
        atol=1e-14,
        rtol=1e-14,
    )


def test_generated_reference_is_periodic_and_matches_initial_condition(tmp_path):
    cache_path = tmp_path / "kdv-small.pt"
    t, x, solution = kdv_reference(
        DEFAULT_MU_SQ,
        modes=31,
        time_steps=5,
        substeps_per_output=8,
        cache_path=str(cache_path),
    )

    assert t.shape == (6,)
    assert x.shape == (32,)
    assert solution.shape == (6, 32)
    assert torch.equal(solution[:, 0], solution[:, -1])
    assert torch.allclose(solution[0], torch.cos(torch.pi * x), atol=1e-14)
    assert torch.isfinite(solution).all()
    assert cache_path.is_file()

    cached = kdv_reference(
        DEFAULT_MU_SQ,
        modes=31,
        time_steps=5,
        substeps_per_output=8,
        cache_path=str(cache_path),
    )
    assert all(torch.equal(a, b) for a, b in zip((t, x, solution), cached))


def test_generated_reference_changes_with_mu_sq(tmp_path):
    kwargs = dict(modes=31, time_steps=5, substeps_per_output=8)
    _, _, low_dispersion = kdv_reference(
        0.015**2, cache_path=str(tmp_path / "low.pt"), **kwargs
    )
    _, _, high_dispersion = kdv_reference(
        0.030**2, cache_path=str(tmp_path / "high.pt"), **kwargs
    )

    assert not torch.allclose(low_dispersion[-1], high_dispersion[-1])


def test_default_reference_reproduces_legacy_jaxpi_solution(tmp_path):
    _, _, solution = kdv_reference(
        DEFAULT_MU_SQ, cache_path=str(tmp_path / "default.pt")
    )
    indices = torch.tensor([0, 64, 128, 256, 384, 511])
    expected_final = torch.tensor(
        [
            -0.3157358614849948,
            0.020054953752303047,
            -0.23590399285921657,
            -0.6404276746524709,
            -0.40932825646744,
            -0.3157358614849948,
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(
        solution[-1, indices], expected_final, atol=2e-8, rtol=2e-8
    )


def test_missing_mat_reference_is_downloaded_once(tmp_path, monkeypatch):
    path = tmp_path / "kdv.mat"
    calls = []

    def fake_download(url, destination):
        calls.append(url)
        t = np.linspace(0.0, 1.0, 3)
        x = np.linspace(-1.0, 1.0, 5)
        savemat(
            destination,
            {"t": t[None, :], "x": x[None, :], "usol": np.outer(t + 1, x)},
        )
        return destination, None

    def fail_if_generated(*_args, **_kwargs):
        raise AssertionError("canonical kdv.mat should be downloaded first")

    monkeypatch.setattr(kdv_module.urllib.request, "urlretrieve", fake_download)
    monkeypatch.setattr(kdv_module, "kdv_reference", fail_if_generated)

    downloaded = load_kdv_reference(str(path))
    cached = load_kdv_reference(str(path))

    assert path.is_file()
    assert calls == [kdv_module.KDV_URL]
    assert all(torch.equal(a, b) for a, b in zip(downloaded, cached))


def test_missing_mat_reference_generates_when_offline(tmp_path, monkeypatch):
    path = tmp_path / "kdv.mat"

    def fail_download(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(kdv_module.urllib.request, "urlretrieve", fail_download)

    generated = load_kdv_reference(str(path))

    assert path.is_file()
    assert tuple(tensor.shape for tensor in generated) == (
        (201,),
        (512,),
        (201, 512),
    )

    def fail_if_regenerated(*_args, **_kwargs):
        raise AssertionError("existing kdv.mat should be reused")

    monkeypatch.setattr(kdv_module, "kdv_reference", fail_if_regenerated)
    cached = load_kdv_reference(str(path))
    assert all(torch.equal(a, b) for a, b in zip(generated, cached))


@pytest.mark.parametrize("mu_sq", [0.0, -1e-4, float("nan"), float("inf")])
def test_generated_reference_rejects_invalid_mu_sq(mu_sq):
    with pytest.raises(ValueError, match="mu_sq"):
        kdv_reference(mu_sq, modes=15, time_steps=2, substeps_per_output=1)


def test_cli_accepts_mu_sq(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["kdv_pinn", "--optimizer", "adamw", "--mu-sq", "0.0009"],
    )

    args = parse_args()

    assert args.mu_sq == pytest.approx(0.0009)
