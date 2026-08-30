import pytest
import torch

from experiments.resnets.cifar100 import MERGE_DIMS_MODES, build_optimizer


@pytest.mark.parametrize("mode", MERGE_DIMS_MODES)
@pytest.mark.parametrize("optimizer_name", ["gnome_hutchinson", "soap"])
def test_cifar100_passes_merge_dims_to_matrix_optimizers(mode, optimizer_name):
    parameter = torch.nn.Parameter(torch.zeros(2, 3, 2, 2))

    optimizer, config = build_optimizer(
        optimizer_name,
        [parameter],
        lr=1e-3,
        weight_decay=0.0,
        merge_dims=mode,
    )

    assert optimizer.defaults["merge_dims"] == MERGE_DIMS_MODES[mode]
    assert config["merge_dims"] == MERGE_DIMS_MODES[mode]
    assert config["merge_dims_mode"] == mode


def test_cifar100_merge_dims_does_not_affect_adamw():
    parameter = torch.nn.Parameter(torch.zeros(2, 3, 2, 2))

    optimizer, config = build_optimizer(
        "adamw",
        [parameter],
        lr=1e-3,
        weight_decay=0.0,
        merge_dims="patch",
    )

    assert "merge_dims_mode" not in config


@pytest.mark.parametrize("optimizer_name", ["gnome_fisher", "soap"])
def test_cifar100_rejects_unknown_merge_dims_mode(optimizer_name):
    parameter = torch.nn.Parameter(torch.zeros(2, 3, 2, 2))

    with pytest.raises(ValueError, match="unknown merge-dims mode"):
        build_optimizer(
            optimizer_name,
            [parameter],
            lr=1e-3,
            weight_decay=0.0,
            merge_dims="unknown",
        )
