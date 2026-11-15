"""Translate existing parametrized test cases to Python config overrides.

Legacy option lists are test data only; experiment CLIs reject these flags.
Tests of the actual --config file interface live in test_experiment_configs.py.
"""

from pathlib import Path

from experiments.common.config import ConfigParser


def config_overrides(module, options):
    fields = ConfigParser(module.__file__).schema["fields"]
    result = {}
    index = 0
    while index < len(options):
        flag = options[index]
        index += 1
        name = flag.removeprefix("--").replace("-", "_")
        enabled = True
        if name == "arch" and "arch" not in fields and "model" in fields:
            name = "model"
        if name not in fields and name.startswith("no_"):
            name, enabled = name[3:], False
        if name not in fields:
            raise SystemExit(f"Unknown config key: {name}")
        kind, _, _, nargs = fields[name]
        if kind == "bool":
            result[name] = enabled
        else:
            convert = {"int": int, "float": float, "str": str, "Path": Path}[kind]
            if nargs is not None:
                end = index
                while end < len(options) and not options[end].startswith("--"):
                    end += 1
                result[name] = [convert(value) for value in options[index:end]]
                index = end
            else:
                result[name] = convert(options[index])
                index += 1
    return result
