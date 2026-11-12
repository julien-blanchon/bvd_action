# Validation

`check_api.py` checks MP4 and sampled-frame preprocessing, Torch/NumPy inputs, generic choices, batching, and encoded-feature reuse against the pinned VideoJev checkpoint.

```bash
uv run --no-sync --offline python -m experiments.check_api path/to/video.mp4
```

Historical experiments and the research reference checkout were moved out of the working tree. Their original files are preserved in `../bvd_action_archives/research-before-cleanup-2026-11-12.tar.gz`; [measurements](../docs/measurements.md) records the results used in this repo.
