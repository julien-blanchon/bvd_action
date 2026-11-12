# BVD action filtering

The BVD adapter asks whether a clip shows **substantial, goal-directed physical action in the real world**: a person doing a hands-on task with objects, tools, materials, or the environment. Cooking, repair, assembly, craft, cleaning, building, gardening, and operating equipment are examples. Talking, slides, screens, gaming, static scenes, incidental gestures, and passive product display are excluded. Instructional narration is neither required nor sufficient; the action must be visible. The exact question and two choice strings live in [`bvd_jev/policy.py`](../bvd_jev/policy.py).

The model scores `ACTION` against `NON_ACTION`. The BVD policy maps the resulting `p_action` to a label:

| `p_action` | Label |
| ---: | --- |
| Below 0.35 | `NON_ACTION` |
| 0.35 through 0.65 | `UNCERTAIN` |
| Above 0.65 | `ACTION` |

These are review thresholds, **not calibrated probabilities of ground truth**. Each JSONL result also records sampled frames, visual tokens, source location, timing, and whether native boundary re-scoring ran. A clip-level `ACTION` label does not give action start/end times. Splitting long videos into windows is a separate dataset policy.

## BVD pools and the pilot

[LAION describes](https://projects.laion.ai/bvd/download.html) BVD-V-55M as scene clips from **2.4 million randomly sampled source videos** in BVD-RAW. Scene detection and lightweight filtering choose the clips. BVD-RAW contains about **80 million videos and 13.5 million source-video hours**; the 55M clips are not random moments from those videos.

Our local index sums to **168,117 scene-interval hours** across 55,233,640 BVD-V records. A 10,000-clip *caption-based* pilot estimated **53,000–87,000 candidate action-containing clip hours** across those records, depending on whether uncertain captions are included after a small review. This is a search estimate, not a visually verified action-hour count: selected clips can contain non-action time, and captions can miss action. Scaling by the RAW-to-2.4M source-video ratio suggests **roughly 1.8–2.9 million candidate scene hours** if comparable scene extraction and action yield hold across RAW. RAW includes the BVD-V sources, so the estimates are not additive.

A production yield estimate needs a random visual audit across source videos, durations, categories, and languages, followed by calibration of precision, recall, and the fraction of each accepted clip occupied by action. Use the BVD CLI on already downloaded TAR offsets where available; URL metadata alone does not supply video pixels.
