# Reference solutions

Numerically generated and downloaded benchmark reference solutions are cached
here so every experiment reads the same reproducible artifact. The artifacts
are intentionally gitignored: a fresh clone downloads or generates them on
first use, and subsequent runs reuse the matching local cache.

Closed-form analytic references remain defined in their experiment source, and
training datasets remain under `experiments/data/`.

## Allen–Cahn

The existing `allen_cahn_highres.mat` contains 4,095 periodic Fourier points
plus a duplicated endpoint, at 201 times on `[0, 1]`. A refinement study on
2026-11-07 generated `allen_cahn_highres_8191.mat` and
`allen_cahn_highres_8191_tight.mat`, each with 8,191 periodic points plus the
duplicated endpoint. The original cache is retained; `reference="highres"`
now selects `allen_cahn_highres_8191_tight.mat` for evaluation during training.

Both refined solves use the existing Fourier-collocation/DOP853 solver.
The standard solve uses `rtol=2e-11`, `atol=2e-13`; the tight solve uses
`rtol=2e-13`, `atol=2e-15`. They use the same 201 time samples as the original.
Results and artifact hashes are recorded in
[`allen_cahn_8191_validation.json`](allen_cahn_8191_validation.json).

| Comparison | Relative L2 difference |
| --- | ---: |
| 2,047 vs 4,095 points, excluding initial time | 5.84794e-7 |
| 4,095 vs 8,191 points, excluding initial time | 1.44738e-7 |
| 4,095 vs 8,191 points, times at least 0.1 | 3.78184e-8 |
| Worst individual time slice, 4,095 vs 8,191, at t=0.005 | 1.77828e-6 |
| Standard vs tight integration tolerances at 8,191 points | 4.82010e-14 |

Spatial comparisons interpolate the coarse solution onto the fine periodic
grid with Fourier resampling. Initial time is excluded because its exact
analytic values are available; interpolating its nonsmooth periodic extension
would introduce a separate interpolation error. On the original runner's
201-by-4,096 grid, the difference from the refined tight solution is about
1.55527e-7, using analytic values at initial time and the duplicated endpoint.

The spatial refinement differences shrink by a factor of 4.04. Assuming that
trend continues gives an estimated remaining aggregate relative L2 error of
about 4.76e-8 for the 8,191-point reference. This is an asymptotic estimate,
not a certified error bound or a guarantee for individual early-time slices.
The tolerance check shows negligible sensitivity to the requested integration
tolerances compared with the spatial refinement differences.

For evaluation of an already available model, load the tight artifact with
`allen_cahn_reference(cache_path="experiments/reference_solutions/allen_cahn_highres_8191_tight.mat", reference="highres")`.
The runner's `reference="highres"` setting loads this same tight artifact by
default. If missing, it generates 8,191 points at `rtol=2e-13`, `atol=2e-15`,
with a 4,095-point convergence check. The standalone generator has matching
defaults. Historical 4,095-point results remain tied to the original cache.

## Ginzburg–Landau

`experiments.pinns.generate_ginzburg_landau_reference` generates a complex128
Fourier/ETDRK4 solution on 256² points over `[0,1]`, with maximum step 0.001
and 201 stored frames. Versioned `ginzburg_landau_v1_*.npz` caches contain
`t,x,y,u,v`, actual time step, and checked solver metadata. The cache is shared
across all PINN time horizons. Evaluation uses the full Fourier interpolant in
space and cubic interpolation in time, without extrapolation.

```bash
uv run --extra experiments -m experiments.pinns.generate_ginzburg_landau_reference --validate
```

Default temporal/spatial refinement deltas are below `1.2e-8`; the time
interpolation delta is `7.5e-8` or less. See the
[benchmark documentation](../../docs/ginzburg_landau.md) for the validation
protocol and numerical limitations. No generated arrays are tracked in git.

## Burgers

`experiments.pinns.burgers_pinn.burgers_reference()` generates the solution via
the Cole–Hopf integral with 256-point Gauss–Hermite quadrature. Coordinates and
values default to float64 on the existing 101-time × 1024-space evaluation grid.
Generation takes seconds on CPU and requires the `experiments` dependencies.
To generate or load the default cache without training:

```bash
uv run --extra experiments python -c 'from experiments.pinns.burgers_pinn import burgers_reference; burgers_reference()'
```

The cache name starts with `burgers_cole_hopf_v1_` and includes grid size,
viscosity, quadrature order, and dtype. Matching metadata is required even for
an explicit cache path. Old `burgers_reference_nx*.pt` spectral caches are not
used by this generator.

For `nu = 0.01/pi` and `t` in `[0, 1]`, orders 256 and 512 agree within `1e-14`
absolute error across the default grid. Independent 90-digit Fourier/Bessel
spot checks agree within `1e-14`, including near the shock. These are numerical
validation results, not a rigorous uniform error bound. Other viscosities or
time intervals need separate convergence checks. Regression checks live in
`tests/test_burgers_reference.py`.

The previous 1024-point spectral cache had about `1.3993e-6` relative L2 error
and `1.9391e-5` maximum absolute error. The cached original JAXPI `burgers.mat`
has about `4.0053e-5` relative L2 error against Cole–Hopf on its own grid.

New Burgers runs use the float64 Cole–Hopf reference for `rel_l2`, perform error
arithmetic in float64 on CPU, and log `reference_method`, `reference_dtype`,
quadrature order, and grid size. Model inference retains the training dtype:
pass `float64` for double-precision training and inference. The optional
`rel_l2_jaxpi` still measures against the original JAXPI dataset. Historical
logs retain their previous reference and should not be compared at the
`1e-6` scale without rescoring saved models.

Integral formulation: [Burkardt's Burgers exact solution implementation](https://people.sc.fsu.edu/~jburkardt/py_src/burgers_exact/burgers_exact.py),
which cites Basdevant et al., *Spectral and finite difference solutions of the
Burgers equation* (1986). This generator uses converged high-order quadrature,
rather than that implementation's default order of eight.
