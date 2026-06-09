"""
Stage 4: Lip-Sync Rendering

Real-time lip manipulation using Zen Dub's VAE latent-space approach.
"""

import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from .capture import VideoFrame, FaceROI
from .synthesis import VisemeFrame


# Default location of the working zen-dub / MuseTalk PyTorch checkout on this box.
DEFAULT_MUSETALK_REPO = os.environ.get(
    "ZEN_DUB_REPO", "/home/z/work/zen/gh/zen-dub"
)


@dataclass
class MuseTalkResult:
    """Result of a clip-level MuseTalk repaint."""
    output_video: str          # path to the lip-synced .mp4 (video only, no audio mux)
    ran: bool                  # True if MuseTalk actually executed and produced output
    backend: str               # "musetalk" | "skipped"
    note: str = ""


class MuseTalkRenderer:
    """
    REAL lip-sync via the zen-dub / MuseTalk PyTorch model, run as an external
    process (``python -m scripts.inference``).

    MuseTalk is a *clip-level* model (it consumes the whole source video + the
    new audio, re-encodes every face crop's mouth via its VAE + UNet, and writes
    a repainted video). That does not fit the per-frame ``_modify_mouth`` hook,
    so the file-based pipeline calls ``render_clip()`` here instead.

    This wiring builds the MuseTalk inference YAML, invokes the model in its own
    repo (so its ``musetalk.*`` imports resolve), and returns the output path.

    Requirements to actually RUN (all in the zen-dub repo / its env):
      * model weights under ``<repo>/models`` (run ``download_weights.sh``)
      * deps: torch, diffusers, opencv-python, librosa, mmpose/mmcv, ffmpeg
      * a GPU is strongly recommended (CPU works but is very slow)

    If those are missing, ``render_clip`` raises a clear error; callers decide
    whether to fall back to copying the source frames (no lip-sync).
    """

    def __init__(
        self,
        repo_dir: str = DEFAULT_MUSETALK_REPO,
        version: str = "v15",
        python_bin: Optional[str] = None,
        extra_env: Optional[dict] = None,
    ):
        self.repo_dir = repo_dir
        self.version = version
        self.python_bin = python_bin or os.environ.get("PYTHON", "python3")
        self.extra_env = extra_env or {}

    def _weights_present(self) -> bool:
        models = os.path.join(self.repo_dir, "models")
        unet_v15 = os.path.join(models, "zen-dubV15", "unet.pth")
        unet_v1 = os.path.join(models, "zen-dub", "pytorch_model.bin")
        return os.path.exists(unet_v15) or os.path.exists(unet_v1)

    def render_clip(self, video_path: str, audio_path: str, out_video: str) -> MuseTalkResult:
        """
        Repaint the speaker's mouth in ``video_path`` to match ``audio_path``.

        Invokes the real MuseTalk inference script in ``self.repo_dir``.
        """
        scripts_inf = os.path.join(self.repo_dir, "scripts", "inference.py")
        if not os.path.exists(scripts_inf):
            raise FileNotFoundError(f"MuseTalk inference script not found: {scripts_inf}")
        if not self._weights_present():
            raise FileNotFoundError(
                f"MuseTalk weights missing under {self.repo_dir}/models -- "
                f"run {self.repo_dir}/download_weights.sh first."
            )

        os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
        work = tempfile.mkdtemp(prefix="zendub_musetalk_")
        try:
            # MuseTalk reads a YAML task list: { task_0: {video_path, audio_path, result_name} }
            cfg_path = os.path.join(work, "task.yaml")
            result_name = os.path.basename(out_video)
            with open(cfg_path, "w") as f:
                f.write("task_0:\n")
                f.write(f"  video_path: {os.path.abspath(video_path)}\n")
                f.write(f"  audio_path: {os.path.abspath(audio_path)}\n")
                f.write(f"  result_name: {result_name}\n")
                if self.version != "v15":
                    f.write("  bbox_shift: 0\n")

            result_dir = os.path.join(work, "results")
            if self.version == "v15":
                unet = os.path.join(self.repo_dir, "models", "zen-dubV15", "unet.pth")
                ver_arg = "v15"
            else:
                unet = os.path.join(self.repo_dir, "models", "zen-dub", "pytorch_model.bin")
                ver_arg = "v1"
            unet_config = os.path.join(self.repo_dir, "models", "zen-dubV15", "zen-dub.json")

            cmd = [
                self.python_bin, "-m", "scripts.inference",
                "--inference_config", cfg_path,
                "--result_dir", result_dir,
                "--unet_model_path", unet,
                "--unet_config", unet_config,
                "--version", ver_arg,
            ]
            env = dict(os.environ)
            env.setdefault("PYTHONPATH", self.repo_dir)
            env.update(self.extra_env)

            proc = subprocess.run(
                cmd, cwd=self.repo_dir, env=env,
                capture_output=True, text=True, timeout=3600,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"MuseTalk inference failed (rc={proc.returncode}):\n"
                    f"{proc.stderr[-1500:]}"
                )

            # Find the produced video under result_dir/<version>/.
            produced = None
            for root, _dirs, files in os.walk(result_dir):
                for fn in files:
                    if fn == result_name or (fn.endswith(".mp4") and result_name.split(".")[0] in fn):
                        produced = os.path.join(root, fn)
                        break
            if produced is None:
                raise RuntimeError(f"MuseTalk produced no output under {result_dir}")
            shutil.copyfile(produced, out_video)
            return MuseTalkResult(out_video, ran=True, backend="musetalk")
        finally:
            shutil.rmtree(work, ignore_errors=True)


@dataclass
class RenderResult:
    """Result of rendering a frame."""
    frame: np.ndarray
    latency_ms: float
    face_detected: bool


class FaceProcessor:
    """Extract and blend face regions."""

    def __init__(self, crop_size: Tuple[int, int] = (256, 256)):
        self.crop_size = crop_size

    def extract_face(
        self,
        frame: np.ndarray,
        face_roi: FaceROI,
        padding: float = 0.25
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract face region with padding."""
        h, w = frame.shape[:2]
        x, y, bw, bh = face_roi.bbox

        # Add padding
        pad_w = int(bw * padding)
        pad_h = int(bh * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + bw + pad_w)
        y2 = min(h, y + bh + pad_h)

        # Extract and resize
        face_crop = frame[y1:y2, x1:x2]

        try:
            import cv2
            face_resized = cv2.resize(face_crop, self.crop_size)
        except ImportError:
            face_resized = face_crop

        return face_resized, (x1, y1, x2, y2)

    def blend_face(
        self,
        frame: np.ndarray,
        modified_face: np.ndarray,
        bbox: Tuple[int, int, int, int],
        blend_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Blend modified face back into frame."""
        x1, y1, x2, y2 = bbox
        target_h = y2 - y1
        target_w = x2 - x1

        try:
            import cv2

            # Resize modified face to original size
            resized = cv2.resize(modified_face, (target_w, target_h))

            # Create smooth blend mask if not provided
            if blend_mask is None:
                blend_mask = self._create_blend_mask(target_w, target_h)

            # Blend
            result = frame.copy()
            for c in range(3):
                result[y1:y2, x1:x2, c] = (
                    resized[:, :, c] * blend_mask +
                    frame[y1:y2, x1:x2, c] * (1 - blend_mask)
                ).astype(np.uint8)

            return result

        except ImportError:
            result = frame.copy()
            result[y1:y2, x1:x2] = modified_face[:target_h, :target_w]
            return result

    def _create_blend_mask(self, w: int, h: int, feather: int = 10) -> np.ndarray:
        """Create feathered blend mask."""
        mask = np.ones((h, w), dtype=np.float32)

        # Feather edges
        for i in range(feather):
            alpha = i / feather
            mask[i, :] = alpha
            mask[h - 1 - i, :] = alpha
            mask[:, i] = np.minimum(mask[:, i], alpha)
            mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)

        return mask


class AudioEncoder(nn.Module if nn else object):
    """Encode audio features for lip-sync conditioning."""

    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, output_dim: int = 512):
        if nn is None:
            return
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(x)


class OneStepInpainter(nn.Module if nn else object):
    """Single forward-pass lip modification without iterative denoising."""

    def __init__(
        self,
        in_channels: int = 8,  # 4 face latent + 4 audio condition
        out_channels: int = 4,
        hidden_dims: List[int] = None
    ):
        if nn is None:
            return
        super().__init__()

        hidden_dims = hidden_dims or [256, 512, 512, 256]

        layers = []
        current_dim = in_channels

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(current_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
            ])
            current_dim = hidden_dim

        layers.append(nn.Conv2d(current_dim, out_channels, 1))

        self.generator = nn.Sequential(*layers)

    def forward(
        self,
        face_latent: "torch.Tensor",
        audio_condition: "torch.Tensor",
        mouth_mask: "torch.Tensor"
    ) -> "torch.Tensor":
        """Single-step inpainting."""
        # Expand audio condition to spatial dimensions
        b, c = audio_condition.shape[:2]
        h, w = face_latent.shape[2:]
        audio_spatial = audio_condition.view(b, c, 1, 1).expand(b, c, h, w)

        # Concatenate
        x = torch.cat([face_latent, audio_spatial], dim=1)

        # Generate
        output = self.generator(x)

        # Apply mask - only modify mouth region
        result = face_latent * (1 - mouth_mask) + output * mouth_mask

        return result


class TemporalSmoother:
    """Apply temporal smoothing for frame consistency."""

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)

    def smooth(self, frame: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing."""
        self.buffer.append(frame.astype(np.float32))

        if len(self.buffer) < self.window_size:
            return frame

        # Gaussian-weighted average
        weights = [0.25, 0.5, 0.25]  # For window_size=3
        smoothed = sum(w * f for w, f in zip(weights, self.buffer))

        return smoothed.astype(np.uint8)

    def reset(self):
        """Reset buffer."""
        self.buffer.clear()


class ZenDubRenderer:
    """Complete rendering using Zen Dub VAE approach."""

    def __init__(
        self,
        model_path: str = "zenlm/zen-dub",
        device: str = "auto"
    ):
        self.model_path = model_path
        self.device = device
        self.face_processor = FaceProcessor()
        self.temporal_smoother = TemporalSmoother()

        self.vae = None
        self.inpainter = None
        self.audio_encoder = None

    def load(self):
        """Load renderer models."""
        if torch is None:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models (would load from checkpoint)
        self.audio_encoder = AudioEncoder().to(device)
        self.inpainter = OneStepInpainter().to(device)

        # Set to eval mode
        self.audio_encoder.eval()
        self.inpainter.eval()

    async def render_frame(
        self,
        frame: VideoFrame,
        audio_features: np.ndarray,
        viseme: VisemeFrame
    ) -> RenderResult:
        """Render single frame with lip-sync."""
        start_time = time.time()

        if not frame.faces:
            return RenderResult(
                frame=frame.image,
                latency_ms=(time.time() - start_time) * 1000,
                face_detected=False
            )

        # Extract primary face
        face_roi = frame.faces[0]
        face_crop, bbox = self.face_processor.extract_face(frame.image, face_roi)

        # Modify face (simplified without full model)
        modified_face = self._modify_mouth(face_crop, audio_features, viseme)

        # Blend back
        result = self.face_processor.blend_face(frame.image, modified_face, bbox)

        # Apply temporal smoothing
        smoothed = self.temporal_smoother.smooth(result)

        return RenderResult(
            frame=smoothed,
            latency_ms=(time.time() - start_time) * 1000,
            face_detected=True
        )

    def _modify_mouth(
        self,
        face_crop: np.ndarray,
        audio_features: np.ndarray,
        viseme: VisemeFrame
    ) -> np.ndarray:
        """Modify mouth region based on audio/viseme.

        NOTE: real lip-sync in this project is performed at the *clip* level by
        ``MuseTalkRenderer.render_clip`` (the zen-dub/MuseTalk PyTorch model run
        as an external process), because MuseTalk's VAE+UNet repaint needs the
        whole audio track and per-frame face crops with landmarks -- it does not
        map onto this single-frame streaming hook. The file-based demo therefore
        bypasses this method and calls ``MuseTalkRenderer`` directly.

        This per-frame path is kept for the realtime skeleton and returns the
        crop unchanged (identity) until a true one-step inpainter checkpoint is
        wired here.
        """
        return face_crop

    async def render_batch(
        self,
        frames: List[VideoFrame],
        audio: np.ndarray,
        visemes: List[VisemeFrame]
    ) -> List[RenderResult]:
        """Render batch of frames."""
        results = []

        # Extract audio features for each frame
        num_frames = len(frames)
        audio_chunk_size = len(audio) // max(num_frames, 1)

        for i, frame in enumerate(frames):
            # Get corresponding audio features
            start_idx = i * audio_chunk_size
            end_idx = start_idx + audio_chunk_size
            audio_chunk = audio[start_idx:end_idx]

            # Find corresponding viseme
            viseme = self._find_viseme(frame.timestamp, visemes)

            # Compute simple audio features
            audio_features = self._compute_audio_features(audio_chunk)

            # Render
            result = await self.render_frame(frame, audio_features, viseme)
            results.append(result)

        return results

    def _find_viseme(
        self,
        timestamp: float,
        visemes: List[VisemeFrame]
    ) -> VisemeFrame:
        """Find viseme for timestamp."""
        for viseme in visemes:
            if viseme.start_time <= timestamp <= viseme.end_time:
                return viseme

        # Default silent viseme
        return VisemeFrame(
            viseme='silent',
            start_time=timestamp,
            end_time=timestamp,
            intensity=0.0
        )

    def _compute_audio_features(self, audio: np.ndarray) -> np.ndarray:
        """Compute audio features (mel spectrogram)."""
        try:
            import librosa

            # Compute mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio.astype(np.float32),
                sr=16000,
                n_mels=80,
                n_fft=400,
                hop_length=160
            )

            # Convert to log scale
            mel_db = librosa.power_to_db(mel, ref=np.max)

            return mel_db.mean(axis=1)  # Average over time

        except ImportError:
            # Fallback: simple features
            return np.zeros(80, dtype=np.float32)


class RenderStage:
    """Complete render stage for the pipeline."""

    def __init__(
        self,
        model_path: str = "zenlm/zen-dub",
        device: str = "auto"
    ):
        self.renderer = ZenDubRenderer(model_path, device)

    def load(self):
        """Load renderer."""
        self.renderer.load()

    async def render(
        self,
        frames: List[VideoFrame],
        audio: np.ndarray,
        visemes: List[VisemeFrame]
    ) -> List[np.ndarray]:
        """Render frames with lip-sync."""
        results = await self.renderer.render_batch(frames, audio, visemes)
        return [r.frame for r in results]
