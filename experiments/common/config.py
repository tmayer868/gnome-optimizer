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

from experiments.common.config_schema import SCHEMAS, pinn_optimizer_fields


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


def merge_config_overrides(config, overrides, *, flat_optimizer_overrides=False):
    """Deep merge known keys; optionally route sweep shorthand to the chosen block."""
    result = deepcopy(config)
    if "optimizer" in overrides and "optimizer" in result:
        result["optimizer"] = overrides["optimizer"]

    def merge(target, changes, path="CONFIG"):
        for name, value in changes.items():
            if name not in target:
                raise ValueError(f"Unknown config key {path}[{name!r}]")
            if isinstance(value, Mapping) and isinstance(target[name], dict):
                merge(target[name], value, f"{path}[{name!r}]")
            else:
                target[name] = deepcopy(value)

    changes = deepcopy(dict(overrides))
    if flat_optimizer_overrides and "optimizers" in result:
        selected = result["optimizers"].get(result.get("optimizer"), {})
        for name in list(changes):
            if name not in result and name in selected:
                if name in changes.get("optimizers", {}).get(result["optimizer"], {}):
                    raise ValueError(f"Duplicate optimizer override for {name!r}")
                changes.setdefault("optimizers", {}).setdefault(result["optimizer"], {})[name] = changes.pop(name)
    merge(result, changes)
    return result


class ConfigParser(argparse.ArgumentParser):
    """Load configuration while retaining experiment-specific parser.error checks."""

    def __init__(self, source, description=None):
        self.key = config_key(source)
        self.config_path = default_config_path(source)
        self.schema = SCHEMAS[self.key]
        self.optimizer_fields = pinn_optimizer_fields(self.key)
        super().__init__(description=description or f"Settings: {self.config_path}", allow_abbrev=False)
        self.add_argument("--config", type=Path, default=self.config_path,
                          help="Python file defining the complete CONFIG dictionary")

    def parse_args(self, args=None, namespace=None):
        try:
            if isinstance(args, Mapping):
                values = load_config_dict(self.config_path)
                if self.optimizer_fields:
                    # Retain the flat Python override API used by tests and callers.
                    # Files themselves must use the nested layout.
                    grouped = set().union(*self.optimizer_fields.values())
                    nested = {name: value for name, value in args.items() if name not in grouped}
                    values = self.validate(merge_config_overrides(values, nested))
                    values.update(deepcopy({name: value for name, value in args.items() if name in grouped}))
                    return argparse.Namespace(**self.validate_fields(values, self.schema["fields"]))
                values.update(deepcopy(dict(args)))
            else:
                options = super().parse_args(args, namespace)
                values = load_config_dict(options.config)
            return argparse.Namespace(**self.validate(values))
        except (OSError, ValueError, SyntaxError) as error:
            self.error(str(error))

    def validate(self, values):
        if self.optimizer_fields:
            return self.validate_pinn(values)
        return self.validate_fields(values, self.schema["fields"])

    def validate_pinn(self, values):
        grouped = set().union(*self.optimizer_fields.values())
        top_fields = {name: spec for name, spec in self.schema["fields"].items()
                      if name not in grouped}
        top = {name: value for name, value in values.items() if name != "optimizers"}
        result = self.validate_fields(top, top_fields)
        blocks = values.get("optimizers")
        if not isinstance(blocks, dict) or blocks.keys() != self.optimizer_fields.keys():
            raise ValueError(f"CONFIG['optimizers'] must contain {list(self.optimizer_fields)}")
        checked = {}
        for name, fields in self.optimizer_fields.items():
            checked[name] = self.validate_fields(
                blocks[name], {key: self.schema["fields"][key] for key in fields},
                context=f"CONFIG['optimizers'][{name!r}]",
            )
        # Older training functions still read some unused optimizer attributes
        # (e.g. auxiliary batch sizes for logging). Populate those as well, but
        # always give the selected optimizer's settings precedence.
        for block in checked.values():
            for name, value in block.items():
                result.setdefault(name, value)
        result.update(checked[result["optimizer"]])
        if "engdw_fixed" in result and result["optimizer"] != "engdw":
            result["engdw_fixed"] = False
        return result

    def validate_fields(self, values, fields, context="CONFIG"):
        if not isinstance(values, Mapping):
            raise ValueError(f"{context} must be a dictionary")
        missing, unknown = fields.keys() - values.keys(), values.keys() - fields.keys()
        if missing or unknown:
            raise ValueError(f"{context}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}")
        result = {}
        for name, spec in fields.items():
            kind, nullable, choices, nargs = spec
            value = values[name]
            if value is None:
                if not nullable:
                    raise ValueError(f"{context}[{name!r}] cannot be None")
                result[name] = None
                continue
            items = value if nargs is not None else [value]
            if nargs is not None:
                if not isinstance(value, (list, tuple)):
                    raise ValueError(f"{context}[{name!r}] must be a list or tuple")
                if (nargs == "+" and not value) or (type(nargs) is int and len(value) != nargs):
                    raise ValueError(f"{context}[{name!r}] requires {nargs} values")
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
                    raise ValueError(f"{context}[{name!r}] requires {kind}, got {item!r}")
                if kind in ("int", "float") and not math.isfinite(item):
                    raise ValueError(f"{context}[{name!r}] must be finite")
                if choices is not None and item not in choices:
                    raise ValueError(f"{context}[{name!r}] must be one of {choices}, got {item!r}")
                checked.append(Path(item).expanduser() if kind == "Path" else item)
            result[name] = checked if nargs is not None else checked[0]
        for exclusive in self.schema.get("exclusive", []):
            selected = [name for name in exclusive if result.get(name) is not None and result[name] is not False]
            if len(selected) > 1:
                raise ValueError(f"Mutually exclusive settings: {', '.join(selected)}")
        return result
