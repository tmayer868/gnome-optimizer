"""Settings for experiments.pinns.generate_allen_cahn_reference. Edit CONFIG, then run the module.

None keeps an existing automatic/default-dependent setting where supported.
"""

from pathlib import Path

CONFIG = {
    'modes': 4095,
    # Lower odd resolution used for the convergence report; 0 disables.
    'check_modes': 2047,
    'time_steps': 200,
    'rtol': 2e-11,
    'atol': 2e-13,
    'output': str(Path(__file__).resolve().parents[2] / 'reference_solutions' / 'allen_cahn_highres.mat'),
    # Regenerate and overwrite an existing matching output file.
    'force': False,
}
