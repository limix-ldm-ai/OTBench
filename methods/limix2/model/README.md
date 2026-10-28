# Vendored LimiX inference implementation

This directory contains the LimiX inference implementation used by the benchmark.
Upstream: <https://github.com/limix-ldm/LimiX/>.
The third-party terms are preserved in `LICENSE.txt`; checkpoint terms must be
checked separately when obtaining pretrained weights. No weights or data are
included.

The adapter uses the v2 architecture and the configuration files in `config/`.
Set `LIMIX2_MODEL_PATH` to the separately obtained `LimiX-2.ckpt`, or place it in
`checkpoints/limix2/pretrained/`. `LIMIX2_SOURCE_ROOT` can select an external source
checkout. The benchmark configuration retains its original inference and
sampling settings.

The package requires Python 3.12 or later. Its `constraints.txt` describes a
separate upstream inference environment and pins SciPy 1.16.2; the benchmark
constraints pin SciPy 1.15.2. These environments should not be silently merged.
The model requires a compatible PyTorch/CUDA stack; optional accelerated
operators may additionally require Triton and FlashAttention.

Submission changes are limited to source packaging, removal of developer demos,
checkpoint-conversion and local verification scripts, and portable paths.
`LIMIX_CACHE_DIR` selects the preprocessing cache; otherwise it uses
`$XDG_CACHE_HOME/otbench/limix` or `~/.cache/otbench/limix`.
The source no longer overrides the Hugging Face endpoint.
