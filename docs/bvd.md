# BVD action classification

## What counts as action?

The [BVD question](../bvd_jev/policy.py) asks for **substantial, visible,
goal-directed physical action**. A person should interact with objects, tools,
materials, or the environment. Cooking, repair, assembly, craft, cleaning,
building, gardening, and equipment use are examples.

Talking, slides, screens, gaming, static scenes, and incidental gestures do not
qualify. An instructional title alone is insufficient. A mixed clip can qualify
when it contains a substantial action sequence.

The model scores `ACTION` against `NON_ACTION`. The BVD adapter turns its
`p_action` score into a review label:

| `p_action` | Label |
| ---: | --- |
| `< 0.35` | `NON_ACTION` |
| `0.35–0.65` | `UNCERTAIN` |
| `> 0.65` | `ACTION` |

These thresholds are a filtering policy, not calibrated probabilities of truth.
The label covers the clip as a whole. It does not identify action start and end
times.

## Input and output

The CLI accepts an MP4, a directory, a TAR shard, or a JSON/JSONL manifest.
For clips already stored in BVD TARs, a manifest row can point to the video
bytes directly:

```json
{"id":"clip-1","source_tar":"/path/to/shard.tar","source_byte_offset":123456,"bytes":987654}
```

Each output row includes `label`, `p_action`, `frames`, `visual_tokens`, source
location, stage timing, and whether boundary re-scoring ran. The CLI writes run
metadata beside the JSONL output and checks it before `--resume`.

## What could BVD yield?

[LAION-BVD](https://projects.laion.ai/bvd/download.html) made BVD-V-55M from
scene clips in **2.4 million randomly selected BVD-RAW source videos**. Scene
detection and filtering choose the clips. The source videos are random; the
55 million clips are not random time windows.

| Pool | Available duration | Candidate action-containing clip hours |
| --- | ---: | ---: |
| BVD-V-55M local scene index | 168,117 hours | 53,000–87,000 hours |
| Full BVD-RAW | 13.5 million source-video hours | 1.8–2.9 million hours, extrapolated |

The BVD-V range comes from a 10,000-clip **caption-based** pilot, not a full
visual audit. The RAW range assumes similar yield after comparable scene
extraction across the full pool. RAW includes the BVD-V sources, so the rows are
not additive. Both ranges count *candidate clip duration*, including any
non-action portions inside accepted clips.

Before a large extraction, audit random videos across duration, category, and
language. Measure precision, recall, and the fraction of accepted clip time
that actually shows action.
