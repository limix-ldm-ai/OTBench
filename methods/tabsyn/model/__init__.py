"""Vendored original TabSyn implementation (github.com/amazon-science/tabsyn).

Modules:
    src/               — data/metrics utilities (relative-import package)
    vae/model.py       — VAE (tokenizer/transformer/reconstructor)
    model.py           — MLPDiffusion + EDM Model wrapper
    diffusion_utils.py — EDM sampler + RePaint-style latent sample_impute
    latent_utils.py    — split_num_cat_target / recover_data helpers
    utils_train.py     — preprocess/make_dataset (imports made package-relative)
    pipeline.py        — train_vae / train_diffusion (paths parameterized)
"""
