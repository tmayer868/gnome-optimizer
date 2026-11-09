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
@pytest.mark.parametrize("embed", ["none", "periodic", "fourier"])
@pytest.mark.parametrize("fixed,fix_after", [
    (False, None), (True, None), (False, 0), (False, 1),
    (False, 2), (False, 3), (True, 2),
])
def test_training_sampling_and_precision_switch(tmp_path, monkeypatch, optimizer, embed, fixed, fix_after):
    monkeypatch.setattr(ac, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ac, "allen_cahn_reference", lambda **kw: (
        torch.linspace(0, 1, 3), torch.linspace(-1, 1, 3), torch.ones(3, 3),
    ))
    training, validation = [], []
    residuals, terms = ac.stacked_residuals, ac.term_losses

    def capture_residuals(model, batch, use_bc):
        training.append(tuple(t.clone() for t in batch))
        assert use_bc == (embed not in ("periodic", "fourier"))
        return residuals(model, batch, use_bc)

    def capture_terms(model, batch):
        validation.append(tuple(t.clone() for t in batch))
        return terms(model, batch)

    monkeypatch.setattr(ac, "stacked_residuals", capture_residuals)
    monkeypatch.setattr(ac, "term_losses", capture_terms)
    overrides = config_overrides(ac, [
        "--optimizer", optimizer, "--embed", embed, "--hidden", "4", "--depth", "3",
        "--steps", "4", "--float64-after", "2", "--log-every", "1",
        "--n-pde", "5", "--n-ic", "3", "--n-bc", "1",
        "--runs-dir", str(tmp_path), "--quiet",
    ])
    overrides.update(fixed_dataset=fixed, fix_after=fix_after, float64=False,
                     embed_dim=4, embed_scale=1.0,
                     aux_frac=0.25, reference="jaxpi")
    args = ac.parse_args(overrides)
    with open(ac.train(args)) as stream:
        records = [json.loads(line) for line in stream]

    main_batches = training[::2] if optimizer == "gnome" else training
    fixed_from = 0 if fixed else fix_after
    assert len(main_batches) == 4
    assert [batch[0].dtype for batch in main_batches] == [
        torch.float32, torch.float32, torch.float64, torch.float64,
    ]
    for step in range(1, len(main_batches)):
        for previous, current in zip(main_batches[step - 1], main_batches[step]):
            assert torch.equal(previous.double(), current.double()) == (
                fixed_from is not None and step > fixed_from
            )
    assert len(validation) == 4
    assert all(not torch.equal(previous[0], current[0])
               for previous, current in zip(validation, validation[1:]))
    for main, val in zip(main_batches, validation):
        assert not torch.equal(main[0], val[0])
    if optimizer == "gnome":
        for step, (main, aux) in enumerate(zip(main_batches, training[1::2])):
            is_fixed = fixed_from is not None and step >= fixed_from
            assert tuple(len(t) for t in aux) == ((5, 5, 2, 1) if is_fixed else (8, 8, 2, 2))
            if is_fixed:
                for points, subset in zip(main, aux):
                    assert torch.isin(subset, points).all()
                    assert len(subset.unique()) == len(subset)
            else:
                assert not torch.isin(aux[0], main[0]).all()

    config = records[0]["hyperparameters"]
    assert config["fixed_dataset"] == fixed
    assert config["fix_after"] == fix_after
    assert config["fixed_from_step"] == fixed_from
    assert config["sampling"] == (
        "fixed" if fixed_from == 0 else "resample_then_fixed"
        if fixed_from is not None else "resample_each_step"
    )
    assert config["aux_sampling"] == (
        ("subsample_fixed_main_each_step" if fixed_from == 0 else
         "resample_then_subsample_fixed_main" if fixed_from is not None else "resample_each_step")
        if optimizer == "gnome" else None
    )
    if fixed_from == 0:
        assert (config["n_pde_aux"], config["n_ic_aux"], config["n_bc_aux"]) == (5, 2, 1)
    events = [r for r in records if r["type"] == "sampling"]
    assert [r["step"] for r in events] == ([] if fixed_from is None else [fixed_from])
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


@pytest.mark.parametrize("fix_after", [-1, 4, 5, 1.5, True, "2"])
def test_invalid_fix_after(fix_after):
    with pytest.raises(SystemExit):
        ac.parse_args({"steps": 4, "fix_after": fix_after})


def test_delayed_fixed_dataset_validates_auxiliary_fraction():
    with pytest.raises(SystemExit):
        ac.parse_args({"optimizer": "gnome", "steps": 4, "fix_after": 2,
                       "fixed_dataset": False, "optimizers": {"gnome": {"aux_frac": 1.1}}})
