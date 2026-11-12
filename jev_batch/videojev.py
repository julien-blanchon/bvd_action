"""VideoJev backend: decode/prepare, batched vision, and choice scoring."""

from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from pathlib import Path

from .types import (ChoiceQuestion, EncodedVideo, PreparedVideo,
                    Prediction, ProbabilityGuard, Sampling, VideoFrames, VideoInput)


class VideoJevEngine:
    """Dataset-independent VideoJev inference with the measured 4B fast path.

    Accepts MP4 paths, sampled RGB tensors/arrays, processor-ready patches,
    or pre-encoded vision features. One choice question is shared per batch.
    """

    def __init__(self, model_path: str | Path, *, attention: str = "sdpa",
                 cpu_threads: int = 4, sparse_seek: bool = True,
                 codec_threads: int = 2, fuse_vision_rope: bool = False) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        if cpu_threads < 1 or codec_threads < 1:
            raise ValueError("CPU and codec thread counts must be positive")
        torch.set_num_threads(cpu_threads)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for VideoJev inference")
        self.torch = torch
        self.model_path = str(model_path)
        local_weights = list(Path(self.model_path).glob("*.safetensors"))
        if local_weights:
            minimum = sum(path.stat().st_size for path in local_weights) + (1 << 30)
            free, _ = torch.cuda.mem_get_info(0)
            if free < minimum:
                raise RuntimeError(f"Insufficient free GPU memory: {free / 2**30:.1f} GiB; "
                                   f"need at least {minimum / 2**30:.1f} GiB")
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        # CPU preparation runs on worker threads while this processor scores GPU batches.
        self.video_processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation=attention,
        ).prepare_for_inference()
        self.dense = self.model.config.backbone_config.model_type == "qwen3_5"
        if fuse_vision_rope:
            if not self.dense:
                raise ValueError("Fused vision RoPE is validated only for VideoJev-4B")
            from .vision_rope import install_fused_vision_rope
            install_fused_vision_rope()
        model_package = type(self.model).__module__.rsplit(".", 1)[0]
        preprocessing = importlib.import_module(f"{model_package}.processing_videojev")
        self.pair_timeline = preprocessing.pair_timeline
        self.prepare_video = preprocessing.prepare_video
        self.sparse_seek = sparse_seek
        self.codec_threads = codec_threads

    def prepare(self, source: VideoInput, sampling: Sampling = Sampling()) -> PreparedVideo | EncodedVideo:
        """Prepare a source; tensor inputs are already sampled RGB video frames."""
        if isinstance(source, (PreparedVideo, EncodedVideo)):
            return source
        started = time.perf_counter()
        if isinstance(source, self.torch.Tensor):
            source = VideoFrames(source, fps=sampling.fps)
        else:
            import numpy as np
            if isinstance(source, np.ndarray):
                source = VideoFrames(source, fps=sampling.fps)
        if isinstance(source, VideoFrames):
            from .inputs import prepare_frames
            prepared = prepare_frames(self.video_processor, source, sampling.context,
                                      self.pair_timeline)
            return PreparedVideo(prepared.pixels, prepared.record,
                                 time.perf_counter() - started)
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.sparse_seek or self.codec_threads != 2:
            from .decode import decode_path
            from .inputs import prepare_frames
            frames = decode_path(path, sampling.fps, sampling.max_frames,
                                 codec_threads=self.codec_threads,
                                 sparse_seek=self.sparse_seek)
            prepared = prepare_frames(self.video_processor, frames, sampling.context,
                                      self.pair_timeline)
            return PreparedVideo(prepared.pixels, prepared.record,
                                 time.perf_counter() - started)
        pixels, record = self.prepare_video(self.video_processor, path,
                                            fps=sampling.fps,
                                            max_frames=sampling.max_frames,
                                            context=sampling.context)
        return PreparedVideo(pixels, record, time.perf_counter() - started)

    def encode(self, prepared: list[PreparedVideo]) -> list[EncodedVideo]:
        """Encode once, allowing later questions to reuse vision features."""
        from .vision import encode_many_videos

        features = encode_many_videos(self.model, prepared,
                                      cached_metadata=self.dense)
        return [EncodedVideo(feature, item.record)
                for item, feature in zip(prepared, features)]

    def _native_score(self, encoded: EncodedVideo, question: ChoiceQuestion,
                      context: int) -> list[float]:
        prediction = self.model.predict_features(
            self.processor, encoded.features, encoded.record,
            [question.native()], context=context,
        )[0]
        return [float(value) for value in prediction["probabilities"]]

    def predict_prepared_batch(
        self, inputs: list[PreparedVideo | EncodedVideo], question: ChoiceQuestion,
        sampling: Sampling = Sampling(), *, decision_batch_size: int = 8,
        guard: ProbabilityGuard | None = None,
    ) -> list[Prediction]:
        """Score one choice question for a bounded batch with optional exact guard."""
        if not inputs:
            return []
        if decision_batch_size < 1:
            raise ValueError("decision_batch_size must be positive")
        if guard is not None and guard.option_index >= len(question.options):
            raise ValueError("Guard option index exceeds the question's options")
        from .decision import score_batch, score_packed

        to_encode = [item for item in inputs if isinstance(item, PreparedVideo)]
        started = time.perf_counter()
        encoded_new = iter(self.encode(to_encode))
        self.torch.cuda.synchronize()
        vision_seconds = time.perf_counter() - started if to_encode else 0.0
        encoded = [next(encoded_new) if isinstance(item, PreparedVideo) else item
                   for item in inputs]
        vision_share = vision_seconds / len(to_encode) if to_encode else 0.0
        result: list[Prediction] = []
        for start in range(0, len(inputs), decision_batch_size):
            subset = encoded[start:start + decision_batch_size]
            subset_inputs = inputs[start:start + decision_batch_size]
            records = [item.record for item in subset]
            features = [item.features for item in subset]
            if len(subset) == 1:
                decision_started = time.perf_counter()
                scores = [self._native_score(subset[0], question, sampling.context)]
                self.torch.cuda.synchronize()
                decision_seconds = time.perf_counter() - decision_started
                mode = "native"
            elif self.dense:
                scores, decision_seconds = score_packed(
                    self.model, self.processor, records, features, sampling.context,
                    question,
                )
                mode = "packed_guarded" if guard else "packed"
            else:
                scores, decision_seconds = score_batch(
                    self.model, self.processor, records, features, sampling.context,
                    question,
                )
                mode = "padded_guarded" if guard else "padded"
            raw_scores = [list(pair) for pair in scores]
            fallback = [False] * len(subset)
            if guard is not None and len(subset) > 1:
                guard_started = time.perf_counter()
                for index, pair in enumerate(scores):
                    if guard.matches(pair):
                        scores[index] = self._native_score(
                            subset[index], question, sampling.context)
                        fallback[index] = True
                self.torch.cuda.synchronize()
                decision_seconds += time.perf_counter() - guard_started
            decision_share = decision_seconds / len(subset)
            for source, video, pair, raw, guarded in zip(
                    subset_inputs, subset, scores, raw_scores, fallback):
                probabilities = [float(value) for value in pair]
                prep_seconds = (source.prepare_seconds
                                if isinstance(source, PreparedVideo) else 0.0)
                video_seconds = vision_share if isinstance(source, PreparedVideo) else 0.0
                result.append({
                    "probabilities": probabilities,
                    "answer_index": max(range(len(probabilities)),
                                        key=probabilities.__getitem__),
                    "frames": video.record["frames"],
                    "visual_tokens": video.record["tokens"],
                    "sampled_duration_s": video.record["sampled_duration"],
                    "vision_batch_size": len(to_encode),
                    "decision_batch_size": len(subset),
                    "decision_mode": mode,
                    "native_guard_fallback": guarded,
                    "pre_guard_probabilities": raw,
                    "seconds": {
                        "decode_preprocess": prep_seconds,
                        "vision": video_seconds,
                        "decision": decision_share,
                        "total": prep_seconds + video_seconds + decision_share,
                    },
                })
        return result

    def predict(self, source: VideoInput, question: ChoiceQuestion,
                sampling: Sampling = Sampling(), *,
                guard: ProbabilityGuard | None = None) -> Prediction:
        prepared = self.prepare(source, sampling)
        return self.predict_prepared_batch([prepared], question, sampling,
                                           decision_batch_size=1, guard=guard)[0]

    def predict_questions(
        self, source: VideoInput, questions: Sequence[ChoiceQuestion],
        sampling: Sampling = Sampling(),
    ) -> list[Prediction]:
        """Score several questions over one video with VideoJev's native joint path.

        The visual encoder runs once. Questions that fit the context share a
        language-model forward pass; larger sets are split by VideoJev.
        """
        if not questions:
            raise ValueError("Provide at least one choice question")
        if len({question.id for question in questions}) != len(questions):
            raise ValueError("Choice question IDs must be unique")
        prepared = self.prepare(source, sampling)
        encoded = (prepared if isinstance(prepared, EncodedVideo)
                   else self.encode([prepared])[0])
        return self.model.predict_features(
            self.processor, encoded.features, encoded.record,
            [question.native() for question in questions],
            context=sampling.context,
        )
