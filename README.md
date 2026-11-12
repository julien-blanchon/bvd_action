# Fast video decisions with VideoJev

This project scores predefined answers about videos. Its first use is finding
hands-on action in LAION-BVD, but the inference library accepts other questions
and other video datasets.

- **`jev_batch`** prepares videos and runs batched [VideoJev-4B](https://huggingface.co/ZhuYuhan/VideoJev-4B) choice inference.
- **`bvd_jev`** adds the BVD action question, review thresholds, TAR input, a CLI, and a small UI.

![Video preparation, inference, and BVD labels](docs/assets/pipeline.svg)

## Why a decision model?

Give VideoJev a video, a question, and candidate answers. It scores the candidates
directly. It does not generate a caption or reason through output tokens.

| | Generative video LLM | VideoJev decision |
| --- | --- | --- |
| Output | Generates answer tokens one after another | Scores supplied choices in one forward pass |
| Main work | Video encoding, input prefill, then answer decoding | Video encoding and input prefill, then a small scoring head |
| Good fit | Open-ended descriptions | Known labels and dataset filters |

The expensive video representation can be reused. VideoJev's native **joint** mode
also places several questions after one video prefix when they fit the context.
Extra labels still add question tokens and scoring work. They are **cheaper than
independent full video passes, not free**. Our fastest *multi-video* batch path
currently scores one question at a time; its measured throughput should not be
multiplied by the number of labels. See [architecture](docs/architecture.md).

## Classify BVD action

On the tested GH200, use the existing Python 3.12 environment and downloaded 4B
checkpoint. Replace the paths for another machine:

```bash
export UV_PROJECT_ENVIRONMENT=/e/scratch/reformo/blanchon1/bvd_action_runtime/venv
export VIDEOJEV_MODEL=/e/scratch/reformo/blanchon1/bvd_action_runtime/model
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

uv run --no-sync --offline python -m bvd_jev.cli /path/to/clips \
  --output /path/to/actions.jsonl --fuse-vision-rope
```

`/path/to/clips` may be one MP4, a directory, a TAR shard, or a JSON/JSONL
manifest. For BVD TARs, a manifest row can contain `source_tar`,
`source_byte_offset`, and `bytes`; the videos do not need to be downloaded again
if their TARs are already available. Use `--resume` to continue an output file,
or `--num-shards` and `--shard-index` to run independent GPU processes.

The output contains `p_action` and one of `ACTION`, `NON_ACTION`, or `UNCERTAIN`.
The default review band is 0.35–0.65. The label describes the **whole clip**;
it does not mark action timestamps. See the [BVD policy](docs/bvd.md) for the
exact question and output fields.

For an interactive check, start the UI and open `/ui`:

```bash
uv run --no-sync --offline python -m bvd_jev.serve --host 127.0.0.1 --port 8000
```

## Ask other questions

Use `ChoiceQuestion` for filters such as tool use, indoor/outdoor setting,
visible people, cooking, or instructional demonstrations. For **one question
across many videos**, `BatchRunner` uses the fast batch path:

```python
from pathlib import Path
from jev_batch import BatchRunner, ChoiceQuestion, Item, VideoJevEngine

engine = VideoJevEngine("/path/to/VideoJev-4B")
question = ChoiceQuestion(
    "Does a person visibly use a physical tool?",
    ("YES: a tool is used on an object", "NO: no tool use is visible"),
    id="tool_use",
)
items = (Item(path.name, path) for path in Path("/path/to/clips").glob("*.mp4"))
for item, prediction in BatchRunner(engine, question).run(items):
    print(item.id, prediction["probabilities"])
```

For **several questions about one video**, `predict_questions` uses VideoJev's
native joint mode. It encodes the video once and shares the video prefix when
the questions fit the context:

```python
from bvd_jev.policy import ACTION_QUESTION
from jev_batch import ChoiceQuestion, Sampling, VideoJevEngine

engine = VideoJevEngine("/path/to/VideoJev-4B")
questions = [
    ACTION_QUESTION,
    ChoiceQuestion("Is a hand tool visibly used?", ("YES", "NO"), id="tool"),
    ChoiceQuestion("Is the scene outdoors?", ("YES", "NO"), id="outdoors"),
]
for result in engine.predict_questions("clip.mp4", questions, Sampling(fps=1, max_frames=16)):
    print(result["id"], result["answer"], result["probabilities"])
```

This native joint path is useful for exploring labels. The measured bulk
throughput below is for the single-question batch path; joint multi-question
throughput on a dataset has not been benchmarked here.

## Measured throughput

One local GH200, BVD TAR clips, 1 sampled FPS, at most 16 frames per clip.
Rates include TAR reading, decode, preprocessing, GPU inference, and output.
Model loading is excluded.

| Model and run | Clips | Clips/s | Sampled frames/s |
| --- | ---: | ---: | ---: |
| 4B, optimized holdout | 1,000 | 11.59 | 171 |
| 4B, warm consolidated runs | 300 | 12.06–12.68 | 179–188 |
| 35B-A3B, padded batch | 300 | 5.68 | 84 |

These are historical measurements before the latest typing and documentation
changes. They are not a prediction for GB200 nodes or for extra questions.
[Measurement details](docs/measurements.md) record the settings and limits.

The code is Apache-2.0. Model weights and BVD media keep their own terms;
this repository does not redistribute BVD videos.
