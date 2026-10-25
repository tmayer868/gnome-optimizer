from config_helpers import config_overrides

import json
import math

import pytest
import torch

from experiments.pinns import allen_cahn_pinn as ac


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_auxiliary_subsets_preserve_pairs_and_sample_without_replacement():
    t = torch.arange(32).reshape(-1, 1).float()
    batch = (t, -t, t + 100, t + 200)
    samples = [ac.sample_aux_from_batch(batch, 8, 4, 4) for _ in range(2)]
    for sample in samples:
        torch.testing.assert_close(sample[1], -sample[0])
        for main, aux, size in zip(batch, sample, (8, 8, 4, 4)):
            assert len(aux) == len(aux.unique()) == size
            assert torch.isin(aux, main).all()
    assert not torch.equal(samples[0][0], samples[1][0])
    full = ac.sample_aux_from_batch(batch, 32, 32, 32)
    for main, aux in zip(batch, full):
        torch.testing.assert_close(main.sort(dim=0).values, aux.sort(dim=0).values)


@pytest.mark.parametrize("optimizer", ["gnome", "soap", "adamw"])
@pytest.mark.parametrize("embed", ["none", "periodic"])
@pytest.mark.parametrize("fixed", [False, True])
def test_training_sampling_and_precision_switch(tmp_path, monkeypatch, optimizer, embed, fixed):
    monkeypatch.setattr(ac, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ac, "allen_cahn_reference", lambda **kw: (
        torch.linspace(0, 1, 3), torch.linspace(-1, 1, 3), torch.ones(3, 3),
    ))
    training, validation = [], []
    residuals, terms = ac.stacked_residuals, ac.term_losses

    def capture_residuals(model, batch, use_bc):
        training.append(tuple(t.clone() for t in batch))
        assert use_bc == (embed != "periodic")
        return residuals(model, batch, use_bc)

    def capture_terms(model, batch):
        validation.append(tuple(t.clone() for t in batch))
        return terms(model, batch)

    monkeypatch.setattr(ac, "stacked_residuals", capture_residuals)
    monkeypatch.setattr(ac, "term_losses", capture_terms)
    args = ac.parse_args(config_overrides(ac, [
        "--optimizer", optimizer, "--embed", embed, "--hidden", "4", "--depth", "3",
        "--steps", "3", "--float64-after", "1", "--log-every", "1",
        "--n-pde", "5", "--n-ic", "3", "--n-bc", "1",
        "--runs-dir", str(tmp_path), "--quiet",
    ] + (["--fixed-dataset"] if fixed else [])))
    with open(ac.train(args)) as stream:
        records = [json.loads(line) for line in stream]

    main_batches = training[::2] if optimizer == "gnome" else training
    assert len(main_batches) == 3
    assert [batch[0].dtype for batch in main_batches] == [
        torch.float32, torch.float64, torch.float64,
    ]
    for batch in main_batches[1:]:
        for original, current in zip(main_batches[0], batch):
            assert torch.equal(original.double(), current) == fixed
    assert len(validation) == 3
    assert not torch.equal(validation[0][0], validation[1][0])
    for main, val in zip(main_batches, validation):
        assert not torch.equal(main[0], val[0])
    if fixed and optimizer == "gnome":
        for main, aux in zip(main_batches, training[1::2]):
            for points, subset in zip(main, aux):
                assert torch.isin(subset, points).all()
                assert len(subset.unique()) == len(subset)

    config = records[0]["hyperparameters"]
    assert config["fixed_dataset"] == fixed
    assert config["sampling"] == ("fixed" if fixed else "resample_each_step")
    assert config["aux_sampling"] == (
        ("subsample_fixed_main_each_step" if fixed else "resample_each_step")
        if optimizer == "gnome" else None
    )
    if fixed:
        assert (config["n_pde_aux"], config["n_ic_aux"], config["n_bc_aux"]) == (5, 2, 1)
    assert records[-1]["completed"]
    assert math.isfinite(records[-1]["final_rel_l2"])


@pytest.mark.parametrize("option,value", [
    ("--aux-frac", "0"), ("--aux-frac", "-1"), ("--aux-frac", "1.1"),
    ("--aux-frac", "nan"), ("--aux-frac", "inf"),
    ("--n-pde", "0"), ("--n-ic", "0"), ("--n-bc", "0"),
])
def test_invalid_fixed_dataset_options(option, value):
    with pytest.raises(SystemExit):
        ac.parse_args(config_overrides(ac, ["--optimizer", "gnome", "--fixed-dataset", option, value]))
