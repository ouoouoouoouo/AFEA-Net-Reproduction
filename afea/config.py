"""Hyper-parameters. Values marked PAPER come from Sec. 4.2 / Table 3; everything else is
an assumption documented in ASSUMPTIONS.md."""

DATASET_PRESETS = {
    "iemocap": {
        "num_classes": 4,           # PAPER
        "margin": 1.0,              # PAPER
        "con_weights": (0.8, 0.5, 0.2),  # PAPER (alpha, beta, gamma)
        "num_folds": 5,             # PAPER: five-fold CV (we use leave-one-session-out)
    },
    "ravdess": {
        "num_classes": 8,           # PAPER
        "margin": 1.5,              # PAPER
        "con_weights": (0.3, 0.2, 0.1),  # PAPER
        "num_folds": 5,             # PAPER (we use actor-disjoint folds)
    },
}

TRAIN_DEFAULTS = {
    "batch_size": 64,        # PAPER
    "lr": 1e-3,              # PAPER (Adam, constant)
    "weight_decay": 0.0,     # assumption
    "lstm_hidden": 256,      # assumption: "512 hidden units" = 2 x 256 (D = 512)
    "lstm_dropout": 0.5,     # PAPER
    "afea_layers": 3,        # PAPER
    "fcn_hidden": 512,       # PAPER ("FCN contains 512 neurons")
    "fcn_dropout": 0.0,      # assumption
    "isa_hidden": None,      # assumption: MLP hidden width = D
    "epochs": 50,            # assumption
    "val_ratio": 0.1,        # assumption: stratified hold-out from training folds
    "select": "val",         # assumption: pick epoch with best validation UAR
    "seed": 42,              # assumption
}
