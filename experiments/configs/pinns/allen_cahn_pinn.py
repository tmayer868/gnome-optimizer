"""Settings for experiments.pinns.allen_cahn_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw'.
    'optimizer': 'gnome',
    # Network: plain tanh MLP, the gated modified MLP (Wang et al. 2021), or 'fused' — the plain
    # MLP with [W|b] merged per layer and hidden layers grouped per fuse_every. hidden / depth
    # control all three. Choices: 'mlp', 'modified', 'fused', 'fused-modified'.
    'arch': 'modified',
    # arch fused only: consecutive hidden layers per parameter tensor. 1 = one tensor per layer,
    # the control every larger value is compared against (identical function and initialization,
    # only the grouping differs). 0 (default) fuses the whole stack. Note depth 4 leaves only k=2
    # hidden layers to group, so raise depth to give this room.
    'fuse_every': 0,
    # Input embedding. 'none' feeds raw [t, x] and keeps the soft periodic BC block. 'periodic'
    # feeds [t, cos(pi x), sin(pi x)], making the network exactly period-2 in x — periodicity
    # then holds for every derivative and the BC block is DROPPED (two blocks instead of three).
    # 'polynomial' feeds [t, 1-x^2, x(1-x^2)]; endpoint values match, but derivatives need not,
    # so it keeps the BC block. 'fourier' feeds trainable Fourier features [sin(zB), cos(zB)]
    # over raw z = [t, x]; not exactly periodic, so it KEEPS the BC block. See embed_dim /
    # embed_scale. Choices: 'none', 'periodic', 'polynomial', 'fourier'.
    'embed': 'none',
    # Fourier feature output dim (even). embed fourier.
    'embed_dim': 256,
    # Init bandwidth of the Fourier frequencies: B ~ N(0, scale^2). B is trained, so this is the
    # starting point, not a fixed bandwidth — but a high init still gives a rough surface to
    # start from. Sweep it. embed fourier.
    'embed_scale': 10.0,
    'seed': 0,
    # Allen-Cahn is stiff; may want more than the default.
    'steps': 100000,
    'n_pde': 4000,
    'n_ic': 200,
    'n_bc': 200,
    # Reuse the same PDE/IC/BC training points every step for any optimizer. Gnome draws fresh
    # auxiliary subsets; validation still resamples.
    'fixed_dataset': False,
    # Aux batch sizes for Gnome are max(K_min, int(N * aux_frac)) per block. Each aux pass is a
    # full higher-order residual eval, so keep small. With fixed_dataset, sizes are capped at N
    # and aux_frac must be in (0,1].
    'aux_frac': 0.25,
    'lr': 0.001,
    # Gnome per-coordinate update bound: lambda is set to the smallest value with
    # max|m̂/(v̂+lambda)| <= this, so no coordinate moves more than lr*trust_region in a step.
    # Larger -> weaker bound -> longer steps. 0 disables it, falling back to plain m̂/(v̂+eps)
    # damping.
    'trust_region': 1.0,
    # Gnome curvature-damping epsilon in m̂/(v̂+eps): larger -> more gradient-descent-like,
    # smaller -> fuller Newton step. Allen-Cahn's canyon geometry may want it larger than the
    # 1e-6 default. Gnome only; SOAP/AdamW keep eps=1e-8.
    'eps': 1e-6,
    # First-moment (momentum) EMA for Gnome and SOAP.
    'beta1': 0.99,
    # Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.
    'beta2': 0.999,
    'weight_decay': 1e-8,
    # Network width.
    'hidden': 256,
    # Network depth: Linear-layer count for arch mlp, number of gated hidden layers for arch
    # modified.
    'depth': 4,
    # Linear LR warmup steps, applied to every optimizer.
    'warmup_steps': 200,
    # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero (standard treatment),
    # 1.0 disables decay. Gnome (MSE) never decays regardless.
    'cosine_decay': 0.0,
    # Run in double precision. Allen-Cahn's stiff canyon geometry can push the residual below the
    # float32 floor (~1e-6 relative), where the Gauss-Newton curvature estimate is reading round-
    # off; float64 moves that floor out of the way. MPS has no double support, so this falls back
    # to CPU there.
    'float64': False,
    # Switch to float64 after N float32 steps, preserving optimizer state and LR schedule.
    # Requires 0 <= N < steps; 0 starts in float64. MPS moves to CPU at the switch.
    'float64_after': None,
    # Reference used for rel_l2. 'jaxpi' is the published 201x512 Chebfun table and remains the
    # compatibility default. 'highres' uses
    # experiments/reference_solutions/allen_cahn_highres.mat and generates it automatically when
    # missing. Choices: 'jaxpi', 'highres'.
    'reference': 'jaxpi',
    # Log Gnome's internal state — curvature spectrum, LM damping, trust-region usage — every N
    # steps to a sibling runs/.../{run_id}.diag.jsonl. 0 (default) disables it entirely. Gnome
    # only: SOAP and AdamW expose no such hook.
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log, e.g. '0,4'. Default logs every parameter, which
    # is one record per tensor per logged step — narrow it to keep the file readable.
    'diagnostics_params': None,
    'log_every': 200,
    'runs_dir': 'runs',
    'quiet': False,
}
