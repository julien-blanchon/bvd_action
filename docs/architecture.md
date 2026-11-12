# How the engine works

![Inference pipeline](assets/pipeline.svg)

`jev_batch` is independent of BVD. It accepts a video path, already sampled `VideoFrames` (NumPy or Torch), `PreparedVideo` patches, or `EncodedVideo` features. The caller supplies a `ChoiceQuestion`; the engine returns a probability vector and per-clip timing. Encoded features can be scored again with another question without rerunning the vision encoder.

For file inputs, CPU workers read and sample frames. The default is **1 sampled FPS, at most 16 frames across the clip**; long clips are sampled across their whole duration. A bounded prefetch queue overlaps this work with GPU inference. Prepared clips are grouped by both clip count and visual-token budget, then encoded as a vision batch. The dense 4B path packs choice scoring with variable-length FlashAttention; the 35B MoE path uses padded batches. Results preserve input order, and a failed GPU batch is retried one clip at a time.

The BVD adapter adds TAR-offset reads, an action question, decision thresholds, sharding, resumable JSONL output, and a review UI. For decisions close to a threshold, it reruns the native VideoJev scoring path. This guard protects the **label boundary** when packed scoring differs slightly from native scoring. It is not a model-confidence calibration.

## Code map

| Location | Purpose |
| --- | --- |
| `jev_batch/types.py` | Inputs, questions, sampling, and batch settings |
| `jev_batch/decode.py`, `inputs.py` | MP4 sampling and VideoJev preprocessing |
| `jev_batch/vision.py`, `decision.py` | Batched vision encoding and choice scoring |
| `jev_batch/pipeline.py`, `videojev.py` | Bounded runner and public model interface |
| `bvd_jev/policy.py`, `sources.py` | Action rule and BVD MP4/TAR input |
| `bvd_jev/cli.py`, `serve.py` | Offline CLI and interactive UI |

The implementation uses BF16 and the checkpoint's native model code. GPU token pruning, FP8, and CUDA graphs are research options, not features claimed by this maintained runtime. The inference path favors measured end-to-end throughput on this workload over extra switches that did not produce a reliable gain.
