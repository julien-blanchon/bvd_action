# Validation

`check_api.py` compares MP4 and sampled-frame preprocessing. It also checks
Torch/NumPy inputs, batching, feature reuse, and native joint questions.

```bash
uv run --no-sync --offline python -m experiments.check_api path/to/video.mp4
```

Historical optimization scripts and run records are preserved in the local
verified archive at `../bvd_action_archives/research-before-cleanup-2026-11-12.tar.gz`.
See [measurements](../docs/measurements.md) for the results and their limits.
