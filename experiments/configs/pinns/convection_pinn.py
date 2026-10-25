"""Settings for experiments.pinns.convection_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw', 'engdw'.
    'optimizer': 'gnome',
    # Convection speed. THE knob for this benchmark: the PDE is trivial at every value, but
    # first-order optimizers start failing around 20 and are comprehensively broken by 30-40. The
    # published comparison numbers in the module docstring are for 30-40. Sweep it to draw the
    # failure curve — that curve is the result, more than any single number.
    'beta': 40.0,
    # Network. Defaults to the plain MLP, which is what the published failure results use —
    # switching to 'modified' changes the architecture as well as the optimizer and muddies the
    # comparison. Choices: 'mlp', 'modified', 'fused', 'fused-modified'.
    'arch': 'mlp',
    # arch fused / fused-modified only: consecutive hidden layers per parameter tensor. 1 = one
    # tensor per layer, the control every larger value is compared against. 0 (default) fuses the
    # whole stack.
    'fuse_every': 0,
    # Input embedding. 'none' (default) feeds raw [t, x] and imposes periodicity softly, as a
    # third residual block — this is the published setup. 'periodic' feeds [t, cos x, sin x],
    # making periodicity exact and DROPPING the BC block; note this also hands the network a
    # basis in which sin(x - beta t) is linear, so it is not a like-for-like comparison. Choices:
    # 'none', 'periodic'.
    'embed': 'none',
    'seed': 0,
    'steps': 50000,
    'n_pde': 4000,
    'n_ic': 256,
    'n_bc': 256,
    # Aux batch fraction for Gnome's curvature surrogate.
    'aux_frac': 0.05,
    'lr': 0.001,
    # Gnome trust region: lambda is set to the smallest value with ||m/(v+lambda)||_2 <= this *
    # sqrt(P). 0 disables the solve.
    'trust_region': 1.0,
    # Gnome curvature damping in m/(v+eps). Gnome only.
    'eps': 1e-6,
    # First-moment EMA for Gnome and SOAP.
    'beta1': 0.9,
    # Second-moment / preconditioner EMA for Gnome, SOAP.
    'beta2': 0.99,
    'weight_decay': 1e-8,
    # ENGD-W fixed damping in JJ^T + lambda I; convection starting value, not a tuned result. No
    # weight decay or LR schedule is used.
    'engdw_damping': 1e-6,
    # ENGD-W SPRING-style regularization-center momentum in [0, 1). Zero preserves the original
    # solver; try 0.9, 0.95, or 0.99.
    'engdw_momentum': 0.0,
    # Choose the lowest same-batch loss on 13 log-spaced step sizes from 1e-3 to 1, overriding lr
    # (same grid as Poisson-5D).
    'engdw_line_search': False,
    # ENGD-W only: sample PDE/IC/BC training points once and reuse them every step. Validation
    # points remain independently sampled.
    'engdw_fixed': False,
    # Samples per vectorized Jacobian chunk for ENGD-W; limits differentiation workspace, while
    # the full Jacobian and sample Gram remain dense.
    'engdw_chunk': 32,
    # Network width. Default 50 follows Krishnapriyan.
    'hidden': 50,
    # Linear-layer count for arch mlp/fused (4 hidden layers of width 50 is the published net);
    # gated hidden-layer count for arch modified.
    'depth': 5,
    'warmup_steps': 200,
    # Final-LR fraction for the cosine decay: 0.0 decays to zero, 1.0 disables decay.
    'cosine_decay': 0.0,
    # Run in double precision on CPU. ENGD-W automatically uses float64 on CPU/CUDA without this
    # flag. Forces device=cpu (MPS has no float64) and sets the global default dtype, so params,
    # collocation samples and the analytic reference are all double. Worth having here because
    # the PDE residual carries a factor of beta: at beta=40 any error in u_x is amplified 40x
    # before it reaches the loss, so the residual hits the float32 floor while u itself still
    # looks clean. Costs roughly an order of magnitude in wall time.
    'float64': False,
    # Log Gnome's internal state every N steps to a sibling {run_id}.diag.jsonl. 0 disables.
    # Gnome only.
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log.
    'diagnostics_params': None,
    'log_every': 200,
    'runs_dir': 'runs',
    'quiet': False,
}
