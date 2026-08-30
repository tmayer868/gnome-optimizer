import numpy as np
import pytest
import experiments.pinns.generate_allen_cahn_reference as generator_module

from experiments.pinns.generate_allen_cahn_reference import (
    convergence_diagnostics,
    generate_allen_cahn_reference,
    solve_allen_cahn,
)
from experiments.pinns.allen_cahn_pinn import allen_cahn_reference


def test_spectral_reference_has_expected_periodic_layout_and_initial_data():
    t = np.linspace(0.0, 0.01, 3)
    solution = solve_allen_cahn(15, t, rtol=1e-9, atol=1e-11)

    assert solution.u.shape == (3, 15)
    assert np.array_equal(solution.t, t)
    assert np.allclose(
        solution.u[0], solution.x**2 * np.cos(np.pi * solution.x)
    )
    assert np.isfinite(solution.u).all()


def test_convergence_diagnostics_are_zero_for_identical_solutions():
    solution = solve_allen_cahn(
        15, np.linspace(0.0, 0.1, 3), rtol=1e-9, atol=1e-11
    )

    diagnostics = convergence_diagnostics(solution, solution)

    assert diagnostics["rel_l2_excluding_t0"] < 1e-14
    assert diagnostics["rel_l2_t_ge_0p1"] < 1e-14
    assert diagnostics["max_per_time_rel_l2_excluding_t0"] < 1e-14


def test_generator_reuses_matching_cached_reference(tmp_path, monkeypatch):
    output = tmp_path / "allen_cahn_highres.mat"
    kwargs = dict(
        modes=15,
        check_modes=0,
        time_steps=2,
        rtol=1e-9,
        atol=1e-11,
    )
    generate_allen_cahn_reference(str(output), **kwargs)
    assert output.is_file()

    def fail_if_regenerated(*_args, **_kwargs):
        raise AssertionError("matching Allen-Cahn cache should be reused")

    monkeypatch.setattr(generator_module, "solve_allen_cahn", fail_if_regenerated)
    assert generate_allen_cahn_reference(str(output), **kwargs) == str(output)


def test_highres_loader_generates_missing_reference(tmp_path, monkeypatch):
    output = tmp_path / "allen_cahn_highres.mat"
    original_generator = generator_module.generate_allen_cahn_reference

    def generate_small_reference(path):
        return original_generator(
            path,
            modes=15,
            check_modes=0,
            time_steps=2,
            rtol=1e-9,
            atol=1e-11,
        )

    monkeypatch.setattr(
        generator_module,
        "generate_allen_cahn_reference",
        generate_small_reference,
    )

    t, x, u = allen_cahn_reference(str(output), reference="highres")

    assert output.is_file()
    assert t.shape == (3,)
    assert x.shape == (16,)
    assert u.shape == (3, 16)


@pytest.mark.parametrize("modes", [0, 2, 8])
def test_spectral_reference_rejects_invalid_mode_counts(modes):
    with pytest.raises(ValueError, match="odd integer"):
        solve_allen_cahn(modes, np.linspace(0.0, 0.1, 3))
