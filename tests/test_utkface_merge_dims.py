import pytest
import torch

from experiments.resnets.utkface import MERGE_DIMS_MODES, build_optimizer


@pytest.mark.parametrize("mode", MERGE_DIMS_MODES)
@pytest.mark.parametrize("optimizer_name", ["gnome", "soap"])
def test_utkface_passes_merge_dims_to_matrix_optimizers(mode, optimizer_name):
    parameter = torch.nn.Parameter(torch.zeros(2, 3, 2, 2))

    optimizer, config, _ = build_optimizer(
        optimizer_name,
        [parameter],
        lr=1e-3,
        weight_decay=0.0,
        warmup=0,
        total_steps=10,
        cosine_decay=1.0,
        merge_dims=mode,
    )

    assert optimizer.defaults["merge_dims"] == MERGE_DIMS_MODES[mode]
    assert config["merge_dims"] == MERGE_DIMS_MODES[mode]
    assert config["merge_dims_mode"] == mode


def test_utkface_merge_dims_does_not_affect_adamw():
    parameter = torch.nn.Parameter(torch.zeros(2, 3, 2, 2))

    _, config, _ = build_optimizer(
        "adamw",
        [parameter],
        lr=1e-3,
        weight_decay=0.0,
        warmup=0,
        total_steps=10,
        cosine_decay=1.0,
        merge_dims="patch",
    )

    assert "merge_dims" not in config
    assert "merge_dims_mode" not in config


@pytest.mark.parametrize("optimizer_name", ["gnome", "soap"])
def test_utkface_rejects_unknown_merge_dims_mode(optimizer_name):
    parameter = torch.nn.Parameter(torch.zeros(2, 3, 2, 2))

    with pytest.raises(ValueError, match="unknown merge-dims mode"):
        build_optimizer(
            optimizer_name,
            [parameter],
            lr=1e-3,
            weight_decay=0.0,
            warmup=0,
            total_steps=10,
            cosine_decay=1.0,
            merge_dims="unknown",
        )
