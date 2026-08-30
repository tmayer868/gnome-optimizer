"""Generate a converged spectral reference for the Allen-Cahn benchmark.

The jaxpi reference uses 511 periodic Fourier modes.  That is sufficient for
most PINN comparisons, but its spatial truncation error is visible once the
relative L2 error approaches 1e-5.  This script solves the same PDE with an
independent adaptive time integrator and a finer Fourier grid:

    u_t = 0.0001 u_xx + 5u - 5u^3,  x in [-1, 1),  t in [0, 1].

An odd number of periodic grid points avoids a special Nyquist mode.  The
saved file duplicates the left endpoint at x=1 so it has the same layout as
jaxpi's ``allen_cahn.mat``.

Usage:

    uv run -m experiments.pinns.generate_allen_cahn_reference

The default solve uses 4095 modes and also solves at 2047 modes to report a
spatial convergence delta.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

from experiments.reference_solutions import reference_path


DEFAULT_OUTPUT = reference_path("allen_cahn_highres.mat")


@dataclass(frozen=True)
class SpectralSolution:
    """A periodic solution without the duplicated right endpoint."""

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    nfev: int


def _validate_modes(modes: int, name: str = "modes") -> None:
    if modes < 3 or modes % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= 3; got {modes}")


def solve_allen_cahn(
    modes: int,
    t: np.ndarray,
    *,
    rtol: float = 2e-11,
    atol: float = 2e-13,
) -> SpectralSolution:
    """Solve Allen-Cahn by Fourier collocation and adaptive DOP853."""
    from scipy.integrate import solve_ivp

    _validate_modes(modes)
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or t[0] != 0.0:
        raise ValueError("t must be a 1D grid with at least two points at t=0")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be strictly increasing")

    # Periodic grid on [-1, 1); angular wave numbers are pi times integers.
    x = -1.0 + 2.0 * np.arange(modes, dtype=np.float64) / modes
    wave_number = 2.0 * np.pi * np.fft.fftfreq(modes, d=2.0 / modes)
    diffusion_symbol = -1e-4 * wave_number**2
    # Keep trial steps inside a conservative part of DOP853's stability
    # region.  Without this cap, its initial step-size search can briefly
    # overflow the cubic reaction at very high spatial resolutions before
    # rejecting the trial step.
    max_step = 4.0 / abs(diffusion_symbol.min())
    initial = x**2 * np.cos(np.pi * x)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        u_xx = np.fft.ifft(diffusion_symbol * np.fft.fft(state)).real
        return u_xx + 5.0 * state - 5.0 * state**3

    result = solve_ivp(
        rhs,
        (float(t[0]), float(t[-1])),
        initial,
        method="DOP853",
        t_eval=t,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not result.success:
        raise RuntimeError(f"Allen-Cahn reference solve failed: {result.message}")
    return SpectralSolution(t=t, x=x, u=result.y.T, nfev=result.nfev)


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def convergence_diagnostics(
    fine: SpectralSolution,
    coarse: SpectralSolution,
) -> dict[str, float]:
    """Compare the fine solve with the coarse trigonometric interpolant."""
    from scipy.signal import resample

    if not np.array_equal(fine.t, coarse.t):
        raise ValueError("fine and coarse solutions must use the same time grid")
    # Lift coarse -> fine rather than restricting fine -> coarse. Restriction
    # would discard the high-frequency tail and systematically understate the
    # spatial truncation error we are trying to measure.
    coarse_on_fine = resample(coarse.u, fine.x.size, axis=1)
    diagnostics = {
        # The exact initial condition is known.  Excluding it prevents its
        # non-smooth periodic extension from measuring interpolation error
        # rather than time-evolution error.
        "rel_l2_excluding_t0": relative_l2(
            coarse_on_fine[1:], fine.u[1:]
        ),
    }
    later = np.searchsorted(fine.t, 0.1)
    diagnostics["rel_l2_t_ge_0p1"] = relative_l2(
        coarse_on_fine[later:], fine.u[later:]
    )
    per_time = np.linalg.norm(coarse_on_fine[1:] - fine.u[1:], axis=1)
    per_time /= np.linalg.norm(fine.u[1:], axis=1)
    diagnostics["max_per_time_rel_l2_excluding_t0"] = float(per_time.max())
    return diagnostics


def save_reference(
    path: str,
    solution: SpectralSolution,
    *,
    rtol: float,
    atol: float,
    check_modes: int | None,
    diagnostics: dict[str, float],
) -> None:
    from scipy.io import savemat

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    x_closed = np.linspace(-1.0, 1.0, solution.x.size + 1)
    u_closed = np.concatenate([solution.u, solution.u[:, :1]], axis=1)
    payload: dict[str, object] = {
        "t": solution.t[None, :],
        "x": x_closed[None, :],
        "usol": u_closed,
        "method": "Fourier collocation + scipy DOP853",
        "modes": np.int64(solution.x.size),
        "rtol": np.float64(rtol),
        "atol": np.float64(atol),
        "nfev": np.int64(solution.nfev),
    }
    if check_modes is not None:
        payload["check_modes"] = np.int64(check_modes)
    payload.update({key: np.float64(value) for key, value in diagnostics.items()})
    savemat(path, payload, do_compression=True)


def generate_allen_cahn_reference(
    output: str = DEFAULT_OUTPUT,
    *,
    modes: int = 4095,
    check_modes: int = 2047,
    time_steps: int = 200,
    rtol: float = 2e-11,
    atol: float = 2e-13,
    force: bool = False,
) -> str:
    """Generate the high-resolution reference, or reuse a matching cache."""
    _validate_modes(modes)
    if check_modes:
        _validate_modes(check_modes, "check_modes")
        if check_modes >= modes:
            raise ValueError("check_modes must be smaller than modes")
    if time_steps < 1:
        raise ValueError("time_steps must be >= 1")

    if os.path.isfile(output) and not force:
        from scipy.io import loadmat

        cached = loadmat(output)
        cached_modes = int(np.asarray(cached.get("modes", -1)).squeeze())
        cached_t = np.asarray(cached.get("t", [])).reshape(-1)
        if cached_modes != modes or cached_t.size != time_steps + 1:
            raise ValueError(
                f"Existing reference {output!r} was generated with "
                f"modes={cached_modes}, time_steps={max(0, cached_t.size - 1)}; "
                "choose another output or set force=True to replace it."
            )
        print(
            f"Reusing cached {cached_t.size} x {modes + 1} reference "
            f"from {output}"
        )
        return output

    t = np.linspace(0.0, 1.0, time_steps + 1)
    started = time.perf_counter()
    print(
        f"Solving Allen-Cahn with {modes} periodic modes, "
        f"rtol={rtol:g}, atol={atol:g} ...",
        flush=True,
    )
    fine = solve_allen_cahn(modes, t, rtol=rtol, atol=atol)
    diagnostics: dict[str, float] = {}
    convergence_modes = check_modes or None
    if convergence_modes is not None:
        print(
            f"Solving convergence check at {convergence_modes} modes ...",
            flush=True,
        )
        coarse = solve_allen_cahn(
            convergence_modes, t, rtol=rtol, atol=atol
        )
        diagnostics = convergence_diagnostics(fine, coarse)

    save_reference(
        output,
        fine,
        rtol=rtol,
        atol=atol,
        check_modes=convergence_modes,
        diagnostics=diagnostics,
    )
    elapsed = time.perf_counter() - started
    print(f"Saved {fine.u.shape[0]} x {fine.u.shape[1] + 1} → {output}")
    print(f"Fine solve: nfev={fine.nfev}, elapsed={elapsed:.1f}s")
    for key, value in diagnostics.items():
        print(f"{key}: {value:.6e}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=int, default=4095)
    parser.add_argument(
        "--check-modes",
        type=int,
        default=2047,
        help="Lower odd resolution used for the convergence report; 0 disables.",
    )
    parser.add_argument("--time-steps", type=int, default=200)
    parser.add_argument("--rtol", type=float, default=2e-11)
    parser.add_argument("--atol", type=float, default=2e-13)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate and overwrite an existing matching output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        generate_allen_cahn_reference(
            args.output,
            modes=args.modes,
            check_modes=args.check_modes,
            time_steps=args.time_steps,
            rtol=args.rtol,
            atol=args.atol,
            force=args.force,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
