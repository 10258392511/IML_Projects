from torch.optim import AdamW

configs_train_test_split_ratio = 0.9

configs_normalizer_param = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225)
}

configs_random_crop_resize_param = {
    "size": (224, 224),
    "scale": (0.08, 1.0),
    "ratio": (0.75, 4 / 3)
}

configs_random_affine_param = {
    "degrees": 15,
    "translate": (0.1, 0.1),
    "scale": (0.8, 1.2),
    "shear": 5
}

configs_random_gaussian_noise_param = {
    "std": 0.1
}

configs_color_jitter_param = {
    "brightness": 0.2,
    "contrast": 0.1,
    "saturation": 0,
    "hue": 0.1
}

configs_gaussian_blur_param = {
    "kernel_size": 5,
    "sigma": (0.5, 2)
}

configs_food_taster_param = {
    "resnet_name": "resnet18",
    "feature_dim": 128
}

configs_trainer_param = {
    "opt_args": {
        "class": AdamW,
        "args": {
            "lr": 1e-3
        }
    },
    "alpha": 2  # maximum: 4
}
