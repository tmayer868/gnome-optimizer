"""Settings for experiments.pinns.generate_allen_cahn_reference. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

from pathlib import Path

CONFIG = {
    'modes': 8191,
    # Lower odd resolution used for the convergence report; 0 disables.
    'check_modes': 4095,
    'time_steps': 200,
    'rtol': 2e-13,
    'atol': 2e-15,
    'output': str(Path(__file__).resolve().parents[2] / 'reference_solutions' / 'allen_cahn_highres_8191_tight.mat'),
    # Regenerate and overwrite an existing matching output file.
    'force': False,
}
