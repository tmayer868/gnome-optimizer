"""Python CONFIG files shared by experiment runners.

Command-line entry points accept only --config. In Python, parse_args can
also receive a dictionary of overrides, useful for tests and small sweeps.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path
import pprint
import runpy

from experiments.common.config_schema import SCHEMAS


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"


def config_key(source):
    return Path(source).resolve().relative_to(EXPERIMENTS_DIR).with_suffix("").as_posix()


def default_config_path(source):
    return CONFIGS_DIR / (config_key(source) + ".py")


def load_config_dict(path):
    """Load an independent copy; Python expressions are evaluated on each call."""
    path = Path(path).expanduser().resolve()
    namespace = runpy.run_path(str(path))
    if not isinstance(namespace.get("CONFIG"), dict):
        raise ValueError(f"{path} must define a CONFIG dictionary")
    return deepcopy(namespace["CONFIG"])


def write_config(path, config):
    """Write a standalone config snapshot for a generated run."""
    def portable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [portable(item) for item in value]
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("CONFIG = " + pprint.pformat(portable(config), sort_dicts=False, width=100) + "\n")
    return path


class ConfigParser(argparse.ArgumentParser):
    """Load configuration while retaining experiment-specific parser.error checks."""

    def __init__(self, source, description=None):
        self.key = config_key(source)
        self.config_path = default_config_path(source)
        self.schema = SCHEMAS[self.key]
        super().__init__(description=description or f"Settings: {self.config_path}", allow_abbrev=False)
        self.add_argument("--config", type=Path, default=self.config_path,
                          help="Python file defining the complete CONFIG dictionary")

    def parse_args(self, args=None, namespace=None):
        try:
            if isinstance(args, Mapping):
                values = load_config_dict(self.config_path)
                values.update(deepcopy(dict(args)))
            else:
                options = super().parse_args(args, namespace)
                values = load_config_dict(options.config)
            return argparse.Namespace(**self.validate(values))
        except (OSError, ValueError, SyntaxError) as error:
            self.error(str(error))

    def validate(self, values):
        fields = self.schema["fields"]
        missing, unknown = fields.keys() - values.keys(), values.keys() - fields.keys()
        if missing or unknown:
            raise ValueError(f"CONFIG: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}")
        result = {}
        for name, spec in fields.items():
            kind, nullable, choices, nargs = spec
            value = values[name]
            if value is None:
                if not nullable:
                    raise ValueError(f"CONFIG[{name!r}] cannot be None")
                result[name] = None
                continue
            items = value if nargs is not None else [value]
            if nargs is not None:
                if not isinstance(value, (list, tuple)):
                    raise ValueError(f"CONFIG[{name!r}] must be a list or tuple")
                if (nargs == "+" and not value) or (type(nargs) is int and len(value) != nargs):
                    raise ValueError(f"CONFIG[{name!r}] requires {nargs} values")
            checked = []
            for item in items:
                valid = {
                    "bool": type(item) is bool,
                    "int": type(item) is int,
                    "float": type(item) in (int, float),
                    "str": isinstance(item, str),
                    "Path": isinstance(item, (str, Path)),
                }[kind]
                if not valid:
                    raise ValueError(f"CONFIG[{name!r}] requires {kind}, got {item!r}")
                if kind in ("int", "float") and not math.isfinite(item):
                    raise ValueError(f"CONFIG[{name!r}] must be finite")
                if choices is not None and item not in choices:
                    raise ValueError(f"CONFIG[{name!r}] must be one of {choices}, got {item!r}")
                checked.append(Path(item).expanduser() if kind == "Path" else item)
            result[name] = checked if nargs is not None else checked[0]
        for exclusive in self.schema.get("exclusive", []):
            selected = [name for name in exclusive if result[name] is not None and result[name] is not False]
            if len(selected) > 1:
                raise ValueError(f"Mutually exclusive settings: {', '.join(selected)}")
        return result
