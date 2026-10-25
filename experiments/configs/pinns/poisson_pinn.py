"""Settings for experiments.pinns.poisson_pinn. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

CONFIG = {
    # Choices: 'gnome', 'soap', 'adamw', 'engdw', 'woodbury'.
    'optimizer': 'gnome',
    'seed': 0,
    # Default 50k — Poisson converges much faster than evolution PDEs; usually plateaus within
    # 20-30k.
    'steps': 50000,
    'n_pde': 2000,
    # Boundary points per edge (total BC sample is 4× this).
    'n_bc_per_edge': 50,
    # Reuse the same PDE/BC training points every step for any optimizer. Gnome/Woodbury draw
    # fresh auxiliary subsets of those points; validation still resamples.
    'fixed_dataset': False,
    'aux_frac': 0.05,
    'lr': 0.001,
    # Gnome/Woodbury per-tensor L2 bound: choose damping so ||direction|| <= trust_region *
    # sqrt(||W||^2 + 1). LR scales the bounded direction. 0 disables adaptive damping.
    'trust_region': 1.0,
    # Gnome curvature-damping epsilon in m̂/(v̂+eps): larger -> more gradient-descent-like,
    # smaller -> fuller Newton step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.
    'eps': 1e-6,
    # First-moment EMA: default 0 for Woodbury, 0.9 for Gnome/SOAP.
    'beta1': None,
    # Curvature EMA for Woodbury; second-moment / preconditioner EMA for Gnome/SOAP.
    'beta2': 0.99,
    'weight_decay': 1e-8,
    # Woodbury damping floor; fixed damping when trust_region 0.
    'woodbury_damping': 0.001,
    # Hold parameters fixed while collecting curvature for the first N steps. 0 disables.
    # Independent of the concurrent warmup_steps LR ramp.
    'woodbury_warmup_steps': 100,
    # Woodbury EMA tail tolerance; history=ceil(log(tol)/log(beta2)).
    'history_tol': 0.01,
    # Override Woodbury's derived history length (matrix samples per parameter).
    'max_history': None,
    # Woodbury float64 inner products, solve and reconstruction; CPU on MPS.
    'gram_float64': False,
    # ENGD-W fixed damping in J J^T + lambda I; no weight decay or schedule.
    'engdw_damping': 1e-6,
    # ENGD-W regularization-center momentum in [0,1).
    'engdw_momentum': 0.0,
    # ENGD-W same-batch grid line search over [1e-3,1], overriding lr.
    'engdw_line_search': False,
    # Points per Jacobian chunk; the full Jacobian and sample Gram remain dense.
    'engdw_chunk': 32,
    # ENGD-W-only compatibility flag for fixed_dataset.
    'engdw_fixed': False,
    # MLP width.
    'hidden': 64,
    # MLP depth.
    'depth': 5,
    # Weight grouping — same function class either way, different parameter tensors for the
    # optimizer to precondition over. 'mlp' (default) stores one [W|b] matrix per layer; 'fused'
    # additionally groups hidden layers per fuse_every. Choices: 'mlp', 'fused'.
    'arch': 'mlp',
    # arch fused only: consecutive hidden layers per parameter tensor. 1 = one tensor per layer,
    # which is the control every larger value is compared against (identical function and
    # initialization, only the grouping differs). 0 (default) fuses the whole stack. A short
    # final chunk is fine if it does not divide depth-2. Not just a modelling knob: fewer, bigger
    # tensors cut Gnome's per-tensor per-step overhead, so at modest width raising this is also a
    # speedup (~8% for 1 -> 2 at hidden=64 depth=10). That reverses once the eigenbasis refresh
    # dominates, well above hidden=64.
    'fuse_every': 0,
    # Linear LR warmup for Gnome, SOAP, AdamW, Woodbury; ENGD-W has no schedule.
    'warmup_steps': 200,
    # Final-LR fraction for the baseline cosine decay: 0.0 decays to zero (standard treatment),
    # 1.0 disables decay. Applies to Gnome, SOAP, AdamW, Woodbury; ignored by ENGD-W.
    'cosine_decay': 0.0,
    # Run in double precision on CPU. float32 floors rel_L2 around 1e-5..1e-6; the ENGD high-
    # accuracy benchmark (Müller & Zeinhofer) reaches ~1e-7, which is only reproducible in
    # float64. Forces device=cpu (MPS has no float64) and sets the global default dtype to
    # float64 so params, samples and the reference are all double.
    'float64': False,
    # Log Gnome or Woodbury internal curvature/step diagnostics every N steps to a sibling
    # runs/.../{run_id}.diag.jsonl. 0 (default) disables it entirely. Supported by Gnome and
    # Woodbury.
    'diagnostics_every': 0,
    # Comma-separated parameter indices to log, e.g. '0,4'. Default logs every parameter, which
    # is one record per tensor per logged step — narrow it to keep the file readable.
    'diagnostics_params': None,
    # Every N steps, measure exactly how much curvature the eigenbasis fails to diagonalize
    # (rho), writing runs/.../{run_id}.rho.jsonl. 0 (default) disables. Costs one backward pass
    # per sample (see measure_rho_samples), so keep N large. Gnome only.
    'measure_rho_every': 0,
    # Residual entries per rho measurement. This is the cost knob: one backward pass each.
    'measure_rho_samples': 256,
    # Also compute the best-Kronecker-product error. Builds a (P x P) matrix, so small layers
    # only.
    'measure_rho_kron_floor': False,
    'log_every': 200,
    'runs_dir': 'runs',
    'quiet': False,
}
