# Local performance

![Sampled-frame throughput on one GH200](assets/throughput.svg)

These runs used **one GH200**, BVD TAR clips, **1 sampled FPS**, at most **16
frames per clip**, and **4096 context tokens**. Rates cover TAR reading, CPU
decode/preparation, GPU inference, and result writing. Model load is excluded.
Each run asked **one action question per clip**.

| Run | Clips | Clips/s | Sampled frames/s | Frames counted |
| --- | ---: | ---: | ---: | ---: |
| 4B optimized holdout | 1,000 | 11.59 | 171 | 14,767 |
| 4B warm consolidated | 300 | 12.06–12.68 | 179–188 | 4,444 |
| 35B-A3B padded batch | 300 | 5.68 | 84 | 4,444 |

“Sampled frames/s” means frames sent to the vision model, not all source frames
decoded from the MP4. The 4B holdout took 86.28 seconds; the 35B run took 52.85
seconds. The 300-clip warm runs showed parity with the preceding implementation
on that set. A cold-cache run was slower.

**Limits:** the GPU was shared, and these runs predate the latest typing and
documentation edits. The post-edit GPU API check passed, but sustained throughput
has not been remeasured. The 35B path and clip set do not establish a controlled
4B-versus-35B cost comparison. Joint multi-question throughput, GB200 nodes, and
RAW-video ingestion still need separate measurements.

Original run records remain in the local verified archive
`../bvd_action_archives/research-before-cleanup-2026-11-12.tar.gz`; it is not part
of the public repository. `experiments.check_api` checks input equivalence,
batching, joint questions, and feature reuse. It is a correctness check, not a
throughput benchmark.
