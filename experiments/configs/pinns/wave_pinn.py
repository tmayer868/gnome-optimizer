"""Settings for experiments.pinns.wave_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw'.
    'optimizer': 'gnome',
    # arch fused only: consecutive hidden layers per parameter tensor. 1 = one tensor per layer,
    # the control every larger value is compared against (identical function and initialization,
    # only the grouping differs). 0 (default) fuses the whole stack. depth 4 leaves only k=2
    # hidden layers to group, so raise depth to give this room.
    'fuse_every': 0,
    # Spatial mode number m of the second component: the solution is sin(pi x)cos(c pi t) + A
    # sin(m pi x)cos(c m pi t). Any m solves the same PDE with the same Dirichlet BCs, so this
    # changes ONLY which harmonic sits on top of the fundamental. 4 (default) is jaxpi's
    # benchmark, giving sin(4 pi x)cos(8 pi t); 3 is the Double-PINN paper's, giving sin(3 pi
    # x)cos(6 pi t). Lower m is an easier target — less high-frequency content for the network to
    # resolve — so runs at different m are NOT comparable to each other, only to the paper that
    # uses that m.
    'second_mode': 4.0,
    # Network: plain tanh MLP or the gated modified MLP. Choices: 'mlp', 'modified', 'fused',
    # 'fused-modified'.
    'arch': 'modified',
    # Input embedding. 'fourier' = trainable Fourier features (the spectral aid wave wants).
    # Choices: 'none', 'periodic', 'fourier'.
    'embed': 'fourier',
    # Fourier embedding output dim (even). embed fourier.
    'fourier_dim': 128,
    # Initial Fourier frequency spread (B ~ N(0, scale²)); B is trained with the model. jaxpi
    # uses ~10 for wave. embed fourier.
    'fourier_scale': 8.0,
    # Enforce Dirichlet u=0 by construction via u = sin(πx)·N(t,x); drops the soft bc loss block.
    'hard_bc': False,
    'seed': 0,
    'steps': 20000,
    'n_pde': 4000,
    # IC points (shared by the displacement and velocity IC blocks).
    'n_ic': 200,
    'n_bc': 200,
    # Aux batch sizes for Gnome are max(K_min, int(N * aux_frac)) per block. Each aux pass is a
    # full higher-order residual eval, so keep small.
    'aux_frac': 0.1,
    'lr': 0.001,
    # Gnome per-coordinate update bound: lambda is set to the smallest value with
    # max|m̂/(v̂+lambda)| <= this, so no coordinate moves more than lr*trust_region in a step.
    # Larger -> weaker bound -> longer steps. 0 disables it, falling back to plain m̂/(v̂+eps)
    # damping.
    'trust_region': 1.0,
    # Gnome curvature-damping epsilon in m̂/(v̂+eps). Gnome only; SOAP/AdamW keep eps=1e-8.
    'eps': 1e-6,
    'beta1': 0.9,
    'beta2': 0.99,
    'weight_decay': 1e-8,
    # Network width.
    'hidden': 256,
    # Network depth: Linear-layer count for arch mlp, number of gated hidden layers for arch
    # modified.
    'depth': 4,
    'warmup_steps': 200,
    # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero, 1.0 disables. Gnome
    # never decays.
    'cosine_decay': 0.0,
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
