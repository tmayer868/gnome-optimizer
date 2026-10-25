"""CUB-200 ViT settings. Edit CONFIG, then run experiments.transformers.cub200_vit."""

CONFIG = {
    "optimizer": "gnome",
    "model": {
        "image_size": 64,
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 3,
        "hidden_dim": 192,
        "mlp_dim": 768,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-6,
        "head_init_std": 0.02
    },
    "data": {
        "data_dir": "experiments/data",
        "download": False,
        "augment": True,
        "crop_scale": [0.2, 1.0],
        "crop_ratio": [0.75, 1.3333333333333333],
        "horizontal_flip_p": 0.5,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.1,
            "hue": 0.0
        },
        "eval_resize_ratio": 1.1428571428571428,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225]
    },
    "training": {
        "seed": 0,
        "device": "auto",
        "epochs": 25,
        "batch_size": 128,
        "eval_batch_size": 64,
        "workers": 0,
        "max_steps": None,
        "warmup_steps": 200,
        "min_lr_frac": 0.1,
        "max_grad_norm": None
    },
    "optimizers": {
        "adamw": {
            "lr": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.05
        },
        "soap": {
            "lr": 1e-3,
            "betas": [0.95, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.05,
            "shampoo_beta": 0.95,
            "precondition_frequency": 10,
            "max_precond_dim": 2048,
            "merge_dims": False,
            "precondition_1d": False,
            "normalize_grads": False,
            "correct_bias": True
        },
        "gnome": {
            "lr": 7e-4,
            "betas": [0.95, 0.95],
            "eps": 1e-12,
            "weight_decay": 0.01,
            "shampoo_beta": 0.95,
            "precondition_frequency": 10,
            "max_precond_dim": 2048,
            "merge_dims": False,
            "precondition_1d": False,
            "trust_radius": 1.0,
            "norm_free": False,
            "loss": "cce_hutchinson",
            "aux_batch_size": 5
        },
        "muon": {
            "lr": 5e-3,
            "weight_decay": 0.05,
            "mu": 0.95,
            "adamw": {
                "lr": 3e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-10,
                "weight_decay": 0.01
            }
        }
    },
    "output": {
        "runs_dir": "runs",
        "log_every": 50,
        "eval_every_epochs": 1,
        "save_checkpoint": True,
        "quiet": False
    }
}
