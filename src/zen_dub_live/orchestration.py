"""
Zen-Live Orchestration Layer

Central coordination for real-time dubbing pipeline with broadcast integration.
Integrates with zen-live (github.com/zenlm/zen-live) for WebRTC/broadcast streaming.
"""

import asyncio
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
from enum import Enum

import numpy as np


class OutputProtocol(Enum):
    """Supported output protocols."""
    RTMP = "rtmp"
    SRT = "srt"
    NDI = "ndi"
    SDI = "sdi"
    WEBRTC = "webrtc"
    FILE = "file"


@dataclass
class PipelineMetrics:
    """Real-time pipeline metrics."""
    capture_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    translation_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    synthesis_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    render_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    e2e_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    frames_processed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    def record_capture(self, latency: float):
        self.capture_latencies.append(latency)

    def record_translation(self, latency: float):
        self.translation_latencies.append(latency)

    def record_synthesis(self, latency: float):
        self.synthesis_latencies.append(latency)

    def record_render(self, latency: float):
        self.render_latencies.append(latency)
        self.frames_processed += 1

    def record_e2e(self, latency: float):
        self.e2e_latencies.append(latency)

    def record_error(self):
        self.errors += 1

    def get_summary(self) -> Dict[str, float]:
        """Get latency summary."""
        def percentile(data: deque, p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        uptime = time.time() - self.start_time

        return {
            "capture_p95_ms": percentile(self.capture_latencies, 95) * 1000,
            "translation_p95_ms": percentile(self.translation_latencies, 95) * 1000,
            "synthesis_p95_ms": percentile(self.synthesis_latencies, 95) * 1000,
            "render_p95_ms": percentile(self.render_latencies, 95) * 1000,
            "e2e_p95_ms": percentile(self.e2e_latencies, 95) * 1000,
            "e2e_p99_ms": percentile(self.e2e_latencies, 99) * 1000,
            "fps": self.frames_processed / max(uptime, 0.001),
            "uptime_seconds": uptime,
            "frames_processed": self.frames_processed,
            "errors": self.errors
        }


class OutputAdapter(Protocol):
    """Protocol for output adapters."""

    async def start(self) -> None:
        ...

    async def write(self, audio: np.ndarray, video: np.ndarray) -> None:
        ...

    async def stop(self) -> None:
        ...


class RTMPOutput:
    """RTMP streaming output."""

    def __init__(self, url: str, width: int = 1920, height: int = 1080, fps: int = 30):
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None

    async def start(self):
        """Start RTMP stream."""
        import subprocess

        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '2',
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-c:a', 'aac',
            '-f', 'flv',
            self.url
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

    async def write(self, audio: np.ndarray, video: np.ndarray):
        """Write frame to stream."""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(video.tobytes())
            except BrokenPipeError:
                pass

    async def stop(self):
        """Stop RTMP stream."""
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            self.process = None


class SRTOutput:
    """SRT streaming output."""

    def __init__(self, port: int = 9710, latency_ms: int = 200):
        self.port = port
        self.latency_ms = latency_ms
        self.socket = None

    async def start(self):
        """Start SRT listener."""
        # Would use srt-python library
        pass

    async def write(self, audio: np.ndarray, video: np.ndarray):
        """Write to SRT connection."""
        pass

    async def stop(self):
        """Stop SRT."""
        pass


class NDIOutput:
    """NDI streaming output."""

    def __init__(self, name: str = "Zen-Dub-Live"):
        self.name = name
        self.sender = None

    async def start(self):
        """Start NDI sender."""
        try:
            import ndi
            self.sender = ndi.create_sender(self.name)
        except ImportError:
            pass

    async def write(self, audio: np.ndarray, video: np.ndarray):
        """Write to NDI."""
        if self.sender:
            # Would send NDI frame
            pass

    async def stop(self):
        """Stop NDI."""
        self.sender = None


class FileOutput:
    """File output for testing."""

    def __init__(self, path: str, fps: int = 30):
        self.path = path
        self.fps = fps
        self.writer = None
        self.audio_buffer = []

    async def start(self):
        """Start file writer."""
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (1920, 1080))
        except ImportError:
            pass

    async def write(self, audio: np.ndarray, video: np.ndarray):
        """Write frame to file."""
        if self.writer:
            self.writer.write(video)
        self.audio_buffer.append(audio)

    async def stop(self):
        """Stop and finalize file."""
        if self.writer:
            self.writer.release()
            self.writer = None

        # Save audio
        if self.audio_buffer:
            try:
                import soundfile as sf
                audio_path = self.path.replace('.mp4', '.wav')
                audio = np.concatenate(self.audio_buffer)
                sf.write(audio_path, audio, 24000)
            except ImportError:
                pass


class BroadcastOutput:
    """Factory for broadcast output adapters."""

    @staticmethod
    def create(protocol: OutputProtocol, **kwargs) -> OutputAdapter:
        """Create output adapter by protocol."""
        if protocol == OutputProtocol.RTMP:
            return RTMPOutput(**kwargs)
        elif protocol == OutputProtocol.SRT:
            return SRTOutput(**kwargs)
        elif protocol == OutputProtocol.NDI:
            return NDIOutput(**kwargs)
        elif protocol == OutputProtocol.FILE:
            return FileOutput(**kwargs)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

    @staticmethod
    def create_rtmp(url: str) -> RTMPOutput:
        """Create RTMP output."""
        return RTMPOutput(url)

    @staticmethod
    def create_srt(port: int) -> SRTOutput:
        """Create SRT output."""
        return SRTOutput(port)

    @staticmethod
    def create_ndi(name: str) -> NDIOutput:
        """Create NDI output."""
        return NDIOutput(name)

    @staticmethod
    def create_file(path: str) -> FileOutput:
        """Create file output."""
        return FileOutput(path)


@dataclass
class SessionConfig:
    """Configuration for a dubbing session."""
    session_id: str
    source_language: str = "en"
    target_language: str = "es"
    anchor_voice_id: Optional[str] = None
    output_protocol: OutputProtocol = OutputProtocol.FILE
    output_config: Dict[str, Any] = field(default_factory=dict)


class ZenLiveOrchestrator:
    """
    Central orchestrator for real-time dubbing.

    Integrates with zen-live for WebRTC streaming infrastructure:
    - WebRTC via WHIP/WHEP endpoints
    - Broadcast output (SRT, RTMP, NDI)
    - Control room UI at zen-live /monitor endpoint

    See: https://github.com/zenlm/zen-live
    """

    def __init__(self, zen_live_url: Optional[str] = None):
        self.sessions: Dict[str, "DubbingSession"] = {}
        self.metrics = PipelineMetrics()
        self.zen_live_url = zen_live_url or "http://localhost:8000"

    async def create_session(self, config: SessionConfig) -> "DubbingSession":
        """Create a new dubbing session."""
        session = DubbingSession(config, self.metrics)
        self.sessions[config.session_id] = session
        return session

    async def get_session(self, session_id: str) -> Optional["DubbingSession"]:
        """Get existing session."""
        return self.sessions.get(session_id)

    async def destroy_session(self, session_id: str):
        """Destroy session."""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            await session.stop()

    def get_metrics(self) -> Dict[str, float]:
        """Get global metrics."""
        return self.metrics.get_summary()


# Alias for backwards compatibility
HanzoOrchestrator = ZenLiveOrchestrator


class DubbingSession:
    """A single dubbing session."""

    def __init__(self, config: SessionConfig, metrics: PipelineMetrics):
        self.config = config
        self.metrics = metrics
        self.status = "created"
        self.output: Optional[OutputAdapter] = None

        # Event callbacks
        self.on_translation: Optional[Callable] = None
        self.on_frame: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

    async def start(self):
        """Start the session."""
        # Create output adapter
        self.output = BroadcastOutput.create(
            self.config.output_protocol,
            **self.config.output_config
        )
        await self.output.start()
        self.status = "running"

    async def stop(self):
        """Stop the session."""
        if self.output:
            await self.output.stop()
        self.status = "stopped"

    async def process_segment(
        self,
        audio: np.ndarray,
        video_frames: List[np.ndarray],
        translation: str,
        dubbed_audio: np.ndarray
    ):
        """Process a translated segment."""
        if self.output:
            for frame in video_frames:
                await self.output.write(dubbed_audio, frame)

        if self.on_translation:
            self.on_translation(translation)

        if self.on_frame:
            for frame in video_frames:
                self.on_frame(frame)

    def get_status(self) -> Dict[str, Any]:
        """Get session status."""
        return {
            "session_id": self.config.session_id,
            "status": self.status,
            "source_language": self.config.source_language,
            "target_language": self.config.target_language,
            "output_protocol": self.config.output_protocol.value
        }
