"""
Stage 1: Capture & Segmentation

Audio and video capture with VAD-aware speech segmentation.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np

try:
    import torch
    import silero_vad
except ImportError:
    torch = None
    silero_vad = None


@dataclass
class SpeechSegment:
    """A detected speech segment with timing."""
    audio: np.ndarray
    start_time: float
    end_time: float
    sample_rate: int = 16000

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class FaceROI:
    """Face region of interest with landmarks."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    landmarks: Optional[np.ndarray] = None
    confidence: float = 1.0


@dataclass
class VideoFrame:
    """A single video frame with metadata."""
    image: np.ndarray
    timestamp: float
    faces: List[FaceROI] = field(default_factory=list)
    frame_number: int = 0


class AudioCapture:
    """Real-time audio capture with rolling buffer."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_ms: int = 200,
        lookback_seconds: float = 1.0
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_ms / 1000)
        self.lookback_chunks = int(lookback_seconds * 1000 / chunk_ms)
        self.buffer: deque = deque(maxlen=self.lookback_chunks)
        self.stream = None

    async def start(self, source: str = "default"):
        """Start audio capture from source."""
        # Initialize audio stream (platform-specific)
        try:
            import sounddevice as sd
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size
            )
            self.stream.start()
        except ImportError:
            raise RuntimeError("sounddevice not installed: pip install sounddevice")

    async def capture_chunk(self) -> np.ndarray:
        """Capture a single audio chunk."""
        if self.stream is None:
            raise RuntimeError("Stream not started")

        chunk, _ = self.stream.read(self.chunk_size)
        chunk = chunk.flatten()
        self.buffer.append(chunk)
        return chunk

    def get_lookback(self) -> np.ndarray:
        """Get full lookback buffer."""
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self.buffer))

    async def stop(self):
        """Stop audio capture."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class VADProcessor:
    """Voice Activity Detection with Silero VAD."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        sample_rate: int = 16000
    ):
        self.threshold = threshold
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_ms / 1000)
        self.sample_rate = sample_rate

        self.speech_buffer: List[np.ndarray] = []
        self.silence_samples = 0
        self.is_speaking = False
        self.segment_start = 0.0

        # Load Silero VAD
        if silero_vad is not None:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
        else:
            self.model = None

    def process(self, chunk: np.ndarray, timestamp: float) -> Optional[SpeechSegment]:
        """Process audio chunk and emit speech segments."""
        if self.model is None:
            # Fallback: energy-based VAD
            energy = np.sqrt(np.mean(chunk ** 2))
            confidence = 1.0 if energy > 0.01 else 0.0
        else:
            # Silero VAD
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
            confidence = self.model(chunk_tensor, self.sample_rate).item()

        if confidence > self.threshold:
            # Speech detected
            if not self.is_speaking:
                self.is_speaking = True
                self.segment_start = timestamp - len(chunk) / self.sample_rate

            self.speech_buffer.append(chunk)
            self.silence_samples = 0

        elif self.is_speaking:
            # Potential end of speech
            self.silence_samples += len(chunk)
            self.speech_buffer.append(chunk)  # Include trailing silence

            if self.silence_samples >= self.min_silence_samples:
                # Speech ended
                total_samples = sum(len(c) for c in self.speech_buffer)

                if total_samples >= self.min_speech_samples:
                    segment = SpeechSegment(
                        audio=np.concatenate(self.speech_buffer),
                        start_time=self.segment_start,
                        end_time=timestamp,
                        sample_rate=self.sample_rate
                    )
                    self.speech_buffer = []
                    self.is_speaking = False
                    return segment
                else:
                    # Too short, discard
                    self.speech_buffer = []
                    self.is_speaking = False

        return None


class VideoCapture:
    """Real-time video capture with face detection."""

    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.cap = None
        self.face_detector = None
        self.frame_count = 0

    async def start(self, source: str = "0"):
        """Start video capture from source."""
        try:
            import cv2
            if source.isdigit():
                self.cap = cv2.VideoCapture(int(source))
            else:
                self.cap = cv2.VideoCapture(source)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {source}")

            # Initialize face detector
            try:
                import mediapipe as mp
                self.face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,  # 0 for close-range, 1 for full-range
                    min_detection_confidence=0.5
                )
            except ImportError:
                # Fallback to OpenCV Haar cascades
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )

        except ImportError:
            raise RuntimeError("opencv-python not installed: pip install opencv-python")

    async def capture_frame(self) -> Optional[VideoFrame]:
        """Capture a single frame with face detection."""
        import cv2

        if self.cap is None:
            raise RuntimeError("Video capture not started")

        ret, frame = self.cap.read()
        if not ret:
            return None

        timestamp = time.time()
        self.frame_count += 1

        # Detect faces
        faces = self._detect_faces(frame)

        return VideoFrame(
            image=frame,
            timestamp=timestamp,
            faces=faces,
            frame_number=self.frame_count
        )

    def _detect_faces(self, frame: np.ndarray) -> List[FaceROI]:
        """Detect faces in frame."""
        import cv2

        faces = []
        h, w = frame.shape[:2]

        if hasattr(self.face_detector, 'process'):
            # MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    faces.append(FaceROI(
                        bbox=(x, y, bw, bh),
                        confidence=detection.score[0]
                    ))
        else:
            # OpenCV Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.face_detector.detectMultiScale(gray, 1.1, 4)
            for (x, y, bw, bh) in detections:
                faces.append(FaceROI(bbox=(x, y, bw, bh)))

        return faces

    async def stop(self):
        """Stop video capture."""
        if self.cap:
            self.cap.release()
            self.cap = None


class StreamSynchronizer:
    """Synchronize audio segments with video frames."""

    def __init__(self, max_drift_ms: float = 50):
        self.max_drift = max_drift_ms / 1000
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.video_queue: asyncio.Queue = asyncio.Queue()
        self.video_buffer: List[VideoFrame] = []

    async def add_audio(self, segment: SpeechSegment):
        """Add audio segment to queue."""
        await self.audio_queue.put(segment)

    async def add_video(self, frame: VideoFrame):
        """Add video frame to queue."""
        await self.video_queue.put(frame)

    async def get_synchronized_pair(
        self,
        timeout: float = 1.0
    ) -> Tuple[SpeechSegment, List[VideoFrame]]:
        """Get time-aligned audio segment and video frames."""
        try:
            segment = await asyncio.wait_for(
                self.audio_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError("No audio segment available")

        # Collect frames for this segment's time range
        frames = []

        # Drain existing buffer
        for frame in self.video_buffer:
            if segment.start_time - self.max_drift <= frame.timestamp <= segment.end_time + self.max_drift:
                frames.append(frame)

        # Get new frames
        while True:
            try:
                frame = self.video_queue.get_nowait()
                if frame.timestamp < segment.start_time - self.max_drift:
                    # Old frame, discard
                    continue
                elif frame.timestamp > segment.end_time + self.max_drift:
                    # Future frame, buffer for next segment
                    self.video_buffer.append(frame)
                    break
                else:
                    frames.append(frame)
            except asyncio.QueueEmpty:
                break

        # Sort frames by timestamp
        frames.sort(key=lambda f: f.timestamp)

        return segment, frames


class CaptureStage:
    """Complete capture stage combining audio, video, and synchronization."""

    def __init__(
        self,
        audio_source: str = "default",
        video_source: str = "0",
        sample_rate: int = 16000,
        target_fps: int = 30
    ):
        self.audio_capture = AudioCapture(sample_rate=sample_rate)
        self.video_capture = VideoCapture(target_fps=target_fps)
        self.vad = VADProcessor(sample_rate=sample_rate)
        self.synchronizer = StreamSynchronizer()
        self.audio_source = audio_source
        self.video_source = video_source
        self.running = False

    async def start(self):
        """Start capture stage."""
        await self.audio_capture.start(self.audio_source)
        await self.video_capture.start(self.video_source)
        self.running = True

    async def stop(self):
        """Stop capture stage."""
        self.running = False
        await self.audio_capture.stop()
        await self.video_capture.stop()

    async def stream(self) -> AsyncIterator[Tuple[SpeechSegment, List[VideoFrame]]]:
        """Stream synchronized audio/video pairs."""
        async def audio_task():
            while self.running:
                chunk = await self.audio_capture.capture_chunk()
                segment = self.vad.process(chunk, time.time())
                if segment:
                    await self.synchronizer.add_audio(segment)
                await asyncio.sleep(0)

        async def video_task():
            while self.running:
                frame = await self.video_capture.capture_frame()
                if frame:
                    await self.synchronizer.add_video(frame)
                await asyncio.sleep(self.video_capture.frame_interval)

        # Start capture tasks
        audio_future = asyncio.create_task(audio_task())
        video_future = asyncio.create_task(video_task())

        try:
            while self.running:
                try:
                    pair = await self.synchronizer.get_synchronized_pair()
                    yield pair
                except TimeoutError:
                    continue
        finally:
            audio_future.cancel()
            video_future.cancel()
