"""
Stage 3: Voice Synthesis

Anchor voice synthesis with prosody transfer and viseme generation.
"""

import os
import time
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .engine_client import EngineConfig, SpeechResult, synthesize_speech

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write a float [-1,1] or int16 mono array to a 16-bit PCM wav (stdlib only)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    a = np.asarray(audio)
    if a.dtype != np.int16:
        a = np.clip(a, -1.0, 1.0)
        a = (a * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(a.tobytes())


def _read_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def _placeholder_voice(text: str, sample_rate: int = 24000) -> np.ndarray:
    """
    Deterministic, dependency-free placeholder 'voice'.

    NOTE: this is NOT neural TTS. It is a syllable-paced formant buzz used only
    so the file-based pipeline can produce real audio (and exercise the ffmpeg
    mux) when the gated zen3-tts codec is unavailable. Swap in the real engine
    via HanzoTTSSynthesizer once a TTS server is reachable.
    """
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    if not words:
        words = ["..."]
    # ~3.3 syllables/word, ~5 syllables/sec speaking rate.
    syllables = max(1, sum(max(1, len(w) // 3) for w in words))
    dur = min(30.0, max(0.6, syllables / 5.0))
    n = int(dur * sample_rate)
    t = np.arange(n) / sample_rate
    # Two formants + slowly drifting pitch + per-syllable amplitude envelope.
    f0 = 120.0 + 15.0 * np.sin(2 * np.pi * 0.5 * t)
    sig = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * (2 * f0) * t)
        + 0.12 * np.sin(2 * np.pi * 650 * t)
    )
    syl_rate = syllables / dur
    env = 0.5 * (1.0 + np.sin(2 * np.pi * syl_rate * t - np.pi / 2))
    env = 0.15 + 0.85 * np.clip(env, 0.0, 1.0)
    audio = (sig * env).astype(np.float32)
    audio *= 0.8 / (np.max(np.abs(audio)) + 1e-6)
    return audio


class HanzoTTSSynthesizer:
    """
    REAL speech synthesis via a running hanzo zen3-tts engine.

    Calls ``engine_client.synthesize_speech`` which hits ``POST /v1/audio/speech``
    (or the ``hanzo run speech`` CLI). If ``allow_fallback`` is set and the
    engine is unreachable / the gated codec is unavailable, it falls back to a
    clearly-labeled placeholder voice so the file pipeline still completes.
    """

    def __init__(
        self,
        cfg: Optional[EngineConfig] = None,
        voice: str = "cherry",
        sample_rate: int = 24000,
        allow_fallback: bool = True,
    ):
        self.cfg = cfg
        self.voice = voice
        self.sample_rate = sample_rate
        self.allow_fallback = allow_fallback

    def synthesize_to_file(self, text: str, out_wav: str) -> SpeechResult:
        if self.cfg is not None:
            try:
                return synthesize_speech(
                    self.cfg, text, out_wav, voice=self.voice, sample_rate=self.sample_rate
                )
            except Exception as e:
                if not self.allow_fallback:
                    raise
                note = f"engine TTS unavailable ({type(e).__name__}: {e}); used placeholder voice"
        else:
            note = "no TTS engine configured; used placeholder voice"

        audio = _placeholder_voice(text, self.sample_rate)
        _write_wav(out_wav, audio, self.sample_rate)
        return SpeechResult(out_wav, self.sample_rate, "fallback-formant", note=note)


@dataclass
class ProsodyFeatures:
    """Prosody features extracted from audio."""
    pitch_contour: np.ndarray
    energy_contour: np.ndarray
    duration_ratios: np.ndarray
    tempo: float = 1.0


@dataclass
class VisemeFrame:
    """A single viseme with timing."""
    viseme: str
    start_time: float
    end_time: float
    intensity: float = 1.0


@dataclass
class SynthesisResult:
    """Complete synthesis result."""
    audio: np.ndarray
    sample_rate: int
    visemes: List[VisemeFrame]
    duration: float


class AnchorVoice:
    """Pre-enrolled speaker voice for consistent synthesis."""

    def __init__(
        self,
        voice_id: str,
        reference_audio: Optional[np.ndarray] = None,
        sample_rate: int = 24000
    ):
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self.speaker_embedding = None
        self.tts_model = None

        if reference_audio is not None:
            self.speaker_embedding = self._extract_embedding(reference_audio)

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from reference audio."""
        if torch is None:
            # Return dummy embedding
            return np.random.randn(256).astype(np.float32)

        try:
            from transformers import Wav2Vec2ForXVector

            model = Wav2Vec2ForXVector.from_pretrained("microsoft/wavlm-base-sv")
            model.eval()

            with torch.no_grad():
                audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                outputs = model(audio_tensor)
                embedding = outputs.embeddings.squeeze(0).numpy()

            return embedding / np.linalg.norm(embedding)

        except ImportError:
            # Fallback: random embedding
            return np.random.randn(256).astype(np.float32)

    def load_tts(self, model_path: str = "microsoft/speecht5_tts"):
        """Load TTS model."""
        try:
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

            self.processor = SpeechT5Processor.from_pretrained(model_path)
            self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(model_path)
        except ImportError:
            raise ImportError("transformers not installed")

    async def synthesize(
        self,
        text: str,
        prosody: Optional[ProsodyFeatures] = None
    ) -> np.ndarray:
        """Synthesize speech with anchor voice characteristics."""
        if self.tts_model is None:
            self.load_tts()

        # Process text
        inputs = self.processor(text=text, return_tensors="pt")

        # Use speaker embedding
        speaker_embeddings = torch.tensor(
            self.speaker_embedding, dtype=torch.float32
        ).unsqueeze(0)

        # Generate speech
        with torch.no_grad():
            speech = self.tts_model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings
            )

        audio = speech.numpy()

        # Apply prosody modification if provided
        if prosody is not None:
            audio = self._apply_prosody(audio, prosody)

        return audio

    def _apply_prosody(
        self,
        audio: np.ndarray,
        prosody: ProsodyFeatures,
        strength: float = 0.5
    ) -> np.ndarray:
        """Apply prosody features to audio."""
        # Simple tempo adjustment
        if prosody.tempo != 1.0:
            try:
                import librosa
                audio = librosa.effects.time_stretch(audio, rate=prosody.tempo)
            except ImportError:
                pass

        return audio


class ProsodyTransfer:
    """Transfer prosody from source to target audio."""

    def __init__(self):
        self.pitch_model = None
        self.energy_model = None

    def extract_prosody(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> ProsodyFeatures:
        """Extract prosody features from source audio."""
        try:
            import librosa

            # Pitch extraction
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
            pitch_contour = np.max(pitches, axis=0)

            # Energy extraction
            energy_contour = librosa.feature.rms(y=audio)[0]

            # Duration (placeholder)
            duration_ratios = np.ones(len(pitch_contour))

            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)

            return ProsodyFeatures(
                pitch_contour=pitch_contour,
                energy_contour=energy_contour,
                duration_ratios=duration_ratios,
                tempo=tempo / 120.0  # Normalize to 120 BPM
            )

        except ImportError:
            # Fallback: dummy prosody
            return ProsodyFeatures(
                pitch_contour=np.zeros(100),
                energy_contour=np.ones(100),
                duration_ratios=np.ones(100),
                tempo=1.0
            )

    def apply_prosody(
        self,
        target_audio: np.ndarray,
        source_prosody: ProsodyFeatures,
        strength: float = 0.7
    ) -> np.ndarray:
        """Apply source prosody to target audio."""
        try:
            import librosa
            import soundfile as sf

            # Time-stretch to match tempo
            stretched = librosa.effects.time_stretch(
                target_audio,
                rate=1.0 / (source_prosody.tempo * strength + (1 - strength))
            )

            return stretched

        except ImportError:
            return target_audio


class VisemeGenerator:
    """Generate phoneme-aligned visemes for lip-sync."""

    # Standard viseme mapping (CMU phonemes to visemes)
    PHONEME_TO_VISEME: Dict[str, str] = {
        'AA': 'ah', 'AE': 'ah', 'AH': 'ah',
        'AO': 'oh', 'AW': 'oh',
        'AY': 'ah',
        'B': 'bmp', 'P': 'bmp', 'M': 'bmp',
        'CH': 'sh', 'JH': 'sh', 'SH': 'sh', 'ZH': 'sh',
        'D': 'td', 'T': 'td', 'N': 'td',
        'DH': 'th', 'TH': 'th',
        'EH': 'eh', 'ER': 'eh',
        'EY': 'eh',
        'F': 'fv', 'V': 'fv',
        'G': 'kg', 'K': 'kg', 'NG': 'kg',
        'HH': 'silent',
        'IH': 'ih', 'IY': 'ih',
        'L': 'l',
        'OW': 'oh', 'OY': 'oh',
        'R': 'r',
        'S': 'sz', 'Z': 'sz',
        'UH': 'uw', 'UW': 'uw',
        'W': 'w',
        'Y': 'y'
    }

    def __init__(self):
        self.aligner = None

    def _init_aligner(self):
        """Initialize forced aligner."""
        # Use simple phoneme estimation
        pass

    def generate_visemes(
        self,
        audio: np.ndarray,
        text: str,
        sample_rate: int = 16000
    ) -> List[VisemeFrame]:
        """Generate time-aligned viseme sequence."""
        # Estimate duration
        duration = len(audio) / sample_rate

        # Simple word-based viseme estimation
        words = text.split()
        if not words:
            return [VisemeFrame(
                viseme='silent',
                start_time=0,
                end_time=duration,
                intensity=0.0
            )]

        word_duration = duration / len(words)
        visemes = []

        current_time = 0
        for word in words:
            # Estimate visemes for word
            word_visemes = self._word_to_visemes(word)
            phoneme_duration = word_duration / max(len(word_visemes), 1)

            for viseme in word_visemes:
                visemes.append(VisemeFrame(
                    viseme=viseme,
                    start_time=current_time,
                    end_time=current_time + phoneme_duration,
                    intensity=1.0
                ))
                current_time += phoneme_duration

        return visemes

    def _word_to_visemes(self, word: str) -> List[str]:
        """Convert word to viseme sequence (simplified)."""
        # Simple letter-to-viseme mapping
        letter_visemes = {
            'a': 'ah', 'e': 'eh', 'i': 'ih', 'o': 'oh', 'u': 'uw',
            'b': 'bmp', 'p': 'bmp', 'm': 'bmp',
            'f': 'fv', 'v': 'fv',
            'd': 'td', 't': 'td', 'n': 'td',
            'l': 'l', 'r': 'r',
            's': 'sz', 'z': 'sz',
            'w': 'w', 'y': 'y',
        }

        visemes = []
        for char in word.lower():
            if char in letter_visemes:
                visemes.append(letter_visemes[char])

        return visemes if visemes else ['silent']


class SynthesisStage:
    """Complete synthesis stage for the pipeline."""

    def __init__(
        self,
        anchor_voice_id: str = "default",
        reference_audio: Optional[np.ndarray] = None,
        sample_rate: int = 24000
    ):
        self.anchor_voice = AnchorVoice(
            voice_id=anchor_voice_id,
            reference_audio=reference_audio,
            sample_rate=sample_rate
        )
        self.prosody_transfer = ProsodyTransfer()
        self.viseme_generator = VisemeGenerator()
        self.sample_rate = sample_rate

    async def synthesize(
        self,
        text: str,
        source_audio: Optional[np.ndarray] = None
    ) -> SynthesisResult:
        """Synthesize speech with anchor voice and generate visemes."""
        start_time = time.time()

        # Extract prosody from source if available
        prosody = None
        if source_audio is not None:
            prosody = self.prosody_transfer.extract_prosody(source_audio)

        # Synthesize with anchor voice
        audio = await self.anchor_voice.synthesize(text, prosody)

        # Generate visemes
        visemes = self.viseme_generator.generate_visemes(
            audio, text, self.sample_rate
        )

        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            visemes=visemes,
            duration=time.time() - start_time
        )
