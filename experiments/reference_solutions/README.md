# Reference solutions

Numerically generated and downloaded benchmark reference solutions are cached
here so every experiment reads the same reproducible artifact. The artifacts
are intentionally gitignored: a fresh clone downloads or generates them on
first use, and subsequent runs reuse the matching local cache.

Closed-form analytic references remain defined in their experiment source, and
training datasets remain under `experiments/data/`.

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
pass `--float64` for double-precision training and inference. The optional
`rel_l2_jaxpi` still measures against the original JAXPI dataset. Historical
logs retain their previous reference and should not be compared at the
`1e-6` scale without rescoring saved models.

Integral formulation: [Burkardt's Burgers exact solution implementation](https://people.sc.fsu.edu/~jburkardt/py_src/burgers_exact/burgers_exact.py),
which cites Basdevant et al., *Spectral and finite difference solutions of the
Burgers equation* (1986). This generator uses converged high-order quadrature,
rather than that implementation's default order of eight.
