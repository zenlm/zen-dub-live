"""
Zen-Dub-Live Pipeline

Complete streaming pipeline for real-time dubbing.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import numpy as np

from .capture import CaptureStage, SpeechSegment, VideoFrame
from .translation import TranslationStage, TranslationResult
from .synthesis import SynthesisStage, SynthesisResult
from .render import RenderStage
from .orchestration import (
    BroadcastOutput,
    HanzoOrchestrator,
    OutputProtocol,
    PipelineMetrics,
    SessionConfig
)


@dataclass
class PipelineConfig:
    """Configuration for the dubbing pipeline."""
    # Audio settings
    audio_source: str = "default"
    sample_rate: int = 16000

    # Video settings
    video_source: str = "0"
    target_fps: int = 30

    # Translation settings
    model_path: str = "zenlm/zen-omni"
    source_language: str = "en"
    target_language: str = "es"
    use_speculative: bool = False

    # Synthesis settings
    anchor_voice_id: str = "default"
    synthesis_sample_rate: int = 24000

    # Render settings
    render_model_path: str = "zenlm/zen-dub"

    # Output settings
    output_protocol: OutputProtocol = OutputProtocol.FILE
    output_config: Dict[str, Any] = field(default_factory=dict)

    # Pipeline settings
    max_queue_size: int = 10
    buffer_duration: float = 0.5  # seconds


@dataclass
class DubbedSegment:
    """A fully processed dubbed segment."""
    original_audio: np.ndarray
    translated_text: str
    dubbed_audio: np.ndarray
    original_frames: List[np.ndarray]
    dubbed_frames: List[np.ndarray]
    start_time: float
    end_time: float
    latency: float


class ZenDubLivePipeline:
    """
    Complete real-time dubbing pipeline.

    Pipeline stages:
    1. Capture: Audio/video ingestion with VAD segmentation
    2. Translation: Speech-to-text translation via Zen Omni
    3. Synthesis: Voice synthesis with anchor voice
    4. Render: Lip-sync video generation via Zen Dub

    Target latency: 2.5-3.5 seconds glass-to-glass
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.metrics = PipelineMetrics()

        # Initialize stages
        self.capture = CaptureStage(
            audio_source=self.config.audio_source,
            video_source=self.config.video_source,
            sample_rate=self.config.sample_rate,
            target_fps=self.config.target_fps
        )

        self.translator = TranslationStage(
            model_path=self.config.model_path,
            source_lang=self.config.source_language,
            target_lang=self.config.target_language,
            use_speculative=self.config.use_speculative
        )

        self.synthesizer = SynthesisStage(
            anchor_voice_id=self.config.anchor_voice_id,
            sample_rate=self.config.synthesis_sample_rate
        )

        self.renderer = RenderStage(
            model_path=self.config.render_model_path
        )

        # Inter-stage queues
        self.capture_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        self.translation_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        self.synthesis_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )

        # Output
        self.output = None
        self.running = False

    async def start(self):
        """Start the pipeline."""
        # Load models
        self.translator.load()
        self.renderer.load()

        # Create output adapter
        self.output = BroadcastOutput.create(
            self.config.output_protocol,
            **self.config.output_config
        )
        await self.output.start()

        # Start capture
        await self.capture.start()

        self.running = True

    async def stop(self):
        """Stop the pipeline."""
        self.running = False
        await self.capture.stop()
        if self.output:
            await self.output.stop()

    async def run(self) -> AsyncIterator[DubbedSegment]:
        """Run the pipeline, yielding dubbed segments."""
        await self.start()

        try:
            # Start stage loops
            tasks = [
                asyncio.create_task(self._capture_loop()),
                asyncio.create_task(self._translation_loop()),
                asyncio.create_task(self._synthesis_loop()),
            ]

            # Process and yield results
            async for segment in self._render_loop():
                yield segment

        finally:
            # Cancel tasks
            for task in tasks:
                task.cancel()

            await self.stop()

    async def _capture_loop(self):
        """Capture stage - ingest audio/video."""
        async for segment, frames in self.capture.stream():
            start_time = time.time()
            await self.capture_queue.put((segment, frames, start_time))
            self.metrics.record_capture(segment.duration)

    async def _translation_loop(self):
        """Translation stage - Zen Omni S2S."""
        while self.running:
            try:
                segment, frames, start_time = await asyncio.wait_for(
                    self.capture_queue.get(),
                    timeout=1.0
                )

                trans_start = time.time()
                translation = await self.translator.translate(segment.audio)
                trans_latency = time.time() - trans_start

                await self.translation_queue.put(
                    (translation, segment, frames, start_time)
                )
                self.metrics.record_translation(trans_latency)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.metrics.record_error()
                print(f"Translation error: {e}")

    async def _synthesis_loop(self):
        """Synthesis stage - voice generation."""
        while self.running:
            try:
                translation, segment, frames, start_time = await asyncio.wait_for(
                    self.translation_queue.get(),
                    timeout=1.0
                )

                synth_start = time.time()
                synthesis = await self.synthesizer.synthesize(
                    translation.target_text,
                    segment.audio
                )
                synth_latency = time.time() - synth_start

                await self.synthesis_queue.put(
                    (synthesis, segment, frames, start_time)
                )
                self.metrics.record_synthesis(synth_latency)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.metrics.record_error()
                print(f"Synthesis error: {e}")

    async def _render_loop(self) -> AsyncIterator[DubbedSegment]:
        """Render stage - lip-sync video."""
        while self.running:
            try:
                synthesis, segment, frames, start_time = await asyncio.wait_for(
                    self.synthesis_queue.get(),
                    timeout=1.0
                )

                render_start = time.time()
                dubbed_frames = await self.renderer.render(
                    frames,
                    synthesis.audio,
                    synthesis.visemes
                )
                render_latency = time.time() - render_start

                # Output
                if self.output:
                    for frame in dubbed_frames:
                        await self.output.write(synthesis.audio, frame)

                # Calculate E2E latency
                e2e_latency = time.time() - start_time
                self.metrics.record_render(render_latency)
                self.metrics.record_e2e(e2e_latency)

                yield DubbedSegment(
                    original_audio=segment.audio,
                    translated_text=synthesis.visemes[0].viseme if synthesis.visemes else "",
                    dubbed_audio=synthesis.audio,
                    original_frames=[f.image for f in frames],
                    dubbed_frames=dubbed_frames,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    latency=e2e_latency
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.metrics.record_error()
                print(f"Render error: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Get pipeline metrics."""
        return self.metrics.get_summary()


class SimpleDubber:
    """Simplified interface for file-based dubbing."""

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        model_path: str = "zenlm/zen-omni"
    ):
        self.config = PipelineConfig(
            source_language=source_lang,
            target_language=target_lang,
            model_path=model_path,
            output_protocol=OutputProtocol.FILE
        )

    async def dub_video(
        self,
        input_path: str,
        output_path: str
    ) -> Dict[str, Any]:
        """Dub a video file."""
        self.config.video_source = input_path
        self.config.audio_source = input_path
        self.config.output_config = {"path": output_path}

        pipeline = ZenDubLivePipeline(self.config)

        segments_processed = 0
        async for segment in pipeline.run():
            segments_processed += 1

        return {
            "input": input_path,
            "output": output_path,
            "segments": segments_processed,
            "metrics": pipeline.get_metrics()
        }


async def run_live_dub(
    source_lang: str = "en",
    target_lang: str = "es",
    output_url: str = "rtmp://localhost/live/stream"
):
    """Run live dubbing to RTMP stream."""
    config = PipelineConfig(
        source_language=source_lang,
        target_language=target_lang,
        output_protocol=OutputProtocol.RTMP,
        output_config={"url": output_url}
    )

    pipeline = ZenDubLivePipeline(config)

    print(f"Starting live dub: {source_lang} → {target_lang}")
    print(f"Output: {output_url}")

    async for segment in pipeline.run():
        metrics = pipeline.get_metrics()
        print(f"E2E latency: {metrics['e2e_p95_ms']:.0f}ms | FPS: {metrics['fps']:.1f}")
