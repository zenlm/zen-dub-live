"""
Stage 2: Understanding & Translation

Speech-to-speech translation using Zen Omni with native audio output.

Qwen3-Omni Architecture:
- Thinker: Audio encoder (32L) + Vision encoder (27L) + Text MoE (48L, 128 experts)
- Talker: Text MoE (20L, 128 experts) + Code Predictor (5L)
- Code2Wav: Neural codec (16 quantizers) → 24kHz waveform

Key: Zen Omni does translation AND audio generation end-to-end!
No separate TTS model needed - native speech-to-speech.
"""

import asyncio
import time
from dataclasses import dataclass
from threading import Thread
from typing import AsyncIterator, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoProcessor, TextIteratorStreamer
except ImportError:
    torch = None
    AutoProcessor = None
    TextIteratorStreamer = None


# Built-in Zen Omni speaker voices
# cherry = female, noah = male (originally "Chery" and "Nofish" in base model)
OMNI_SPEAKERS = {
    "cherry": 1,   # Female voice
    "noah": 2,     # Male voice
}


@dataclass
class TranslationChunk:
    """A chunk of translated text with optional audio."""
    text: str
    timestamp: float
    is_final: bool
    confidence: float = 1.0
    audio_chunk: Optional[np.ndarray] = None


@dataclass
class TranslationResult:
    """Complete translation result with native audio output."""
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    duration: float
    # Native audio output from Zen Omni (no separate TTS needed!)
    audio: Optional[np.ndarray] = None
    audio_sample_rate: int = 24000
    speaker_id: str = "cherry"  # Default: female voice


@dataclass
class TranslationPair:
    """A source-target translation pair for context."""
    source: str
    target: str
    speaker_id: str
    timestamp: float


@dataclass
class SpeakerProfile:
    """Speaker profile for voice consistency."""
    speaker_id: str
    name: str
    voice_embedding: Optional[np.ndarray] = None


class TranslationContext:
    """Maintains conversation context for coherent translations."""

    def __init__(self, max_history: int = 10):
        self.history: List[TranslationPair] = []
        self.max_history = max_history
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}

    def add_translation(self, source: str, target: str, speaker_id: str):
        """Add translation pair to context."""
        self.history.append(TranslationPair(
            source=source,
            target=target,
            speaker_id=speaker_id,
            timestamp=time.time()
        ))

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context_prompt(self, num_items: int = 5) -> str:
        """Build context string for translation."""
        lines = []
        for pair in self.history[-num_items:]:
            lines.append(f"[{pair.speaker_id}] {pair.source} -> {pair.target}")
        return "\n".join(lines)

    def register_speaker(self, speaker_id: str, name: str, embedding: Optional[np.ndarray] = None):
        """Register a speaker profile."""
        self.speaker_profiles[speaker_id] = SpeakerProfile(
            speaker_id=speaker_id,
            name=name,
            voice_embedding=embedding
        )

    def get_speaker(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Get speaker profile."""
        return self.speaker_profiles.get(speaker_id)

    def clear(self):
        """Clear context history."""
        self.history.clear()


class ZenOmniTranslator:
    """
    Speech-to-speech translation using Zen Omni with native audio output.

    Zen Omni provides end-to-end translation:
    - Audio input → Thinker (understanding) → Talker (generation) → Code2Wav (audio)
    - Vision input for lip reading and video context
    - Native multilingual speech output (no separate TTS needed)

    Built-in speakers: cherry (female), noah (male)

    Custom voices can be added via voice cloning - see AnchorVoice class.
    """

    def __init__(
        self,
        model_path: str = "zenlm/zen-omni",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        speaker: str = "cherry"  # Default voice (female)
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.speaker = speaker
        self.model = None
        self.processor = None
        self.context = TranslationContext()

    def load(self):
        """Load model and processor."""
        if torch is None:
            raise ImportError("PyTorch not installed: pip install torch")

        from transformers import AutoModelForCausalLM

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype_map.get(self.torch_dtype, torch.bfloat16),
            device_map=self.device,
            trust_remote_code=True,
            enable_audio_output=True  # Enable native audio generation
        )

    def translate(
        self,
        audio: np.ndarray,
        source_lang: str = "en",
        target_lang: str = "es",
        sample_rate: int = 16000,
        video_frames: Optional[List[np.ndarray]] = None,  # For lip reading
        return_audio: bool = True
    ) -> TranslationResult:
        """
        Translate audio with native speech output.

        Args:
            audio: Input audio array
            source_lang: Source language code
            target_lang: Target language code
            sample_rate: Audio sample rate
            video_frames: Optional video frames for lip reading context
            return_audio: Whether to generate native audio output

        Returns:
            TranslationResult with text AND native audio
        """
        if self.model is None:
            self.load()

        start_time = time.time()

        # Build conversation with translation task
        conversation = [
            {
                "role": "system",
                "content": f"You are a translator. Translate speech from {source_lang} to {target_lang}. Respond in {target_lang} audio."
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio}
                ]
            }
        ]

        # Add video context if provided (lip reading)
        if video_frames is not None and len(video_frames) > 0:
            conversation[1]["content"].insert(0, {
                "type": "video",
                "video": video_frames
            })

        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate with audio output
        with torch.no_grad():
            if return_audio and hasattr(self.model, 'generate_with_audio'):
                # Native speech-to-speech generation
                outputs = self.model.generate_with_audio(
                    **inputs,
                    max_new_tokens=512,
                    speaker_id=OMNI_SPEAKERS.get(self.speaker, 2301),
                    do_sample=True,
                    temperature=0.7
                )
                text_output = outputs.text
                audio_output = outputs.audio  # Native 24kHz audio!
            else:
                # Fallback to text-only
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7
                )
                text_output = self.processor.decode(outputs[0], skip_special_tokens=True)
                audio_output = None

        duration = time.time() - start_time

        return TranslationResult(
            source_text="",
            target_text=text_output,
            source_lang=source_lang,
            target_lang=target_lang,
            duration=duration,
            audio=audio_output,  # Native audio from Zen Omni!
            audio_sample_rate=24000,
            speaker_id=self.speaker
        )

    async def translate_stream(
        self,
        audio: np.ndarray,
        source_lang: str = "en",
        target_lang: str = "es",
        sample_rate: int = 16000
    ) -> AsyncIterator[TranslationChunk]:
        """Stream translated text chunks."""
        if self.model is None:
            self.load()

        # Build prompt
        context = self.context.get_context_prompt()
        prompt = f"""Translate speech from {source_lang} to {target_lang}.
Context: {context}
Output:"""

        # Process input
        inputs = self.processor(
            text=prompt,
            audio=audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.model.device)

        # Setup streamer
        streamer = TextIteratorStreamer(
            self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            use_cache=True
        )

        # Run generation in background thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield chunks as they arrive
        collected_text = ""
        for text_chunk in streamer:
            collected_text += text_chunk
            yield TranslationChunk(
                text=text_chunk,
                timestamp=time.time(),
                is_final=False
            )

        thread.join()

        # Final chunk
        yield TranslationChunk(
            text="",
            timestamp=time.time(),
            is_final=True
        )

        # Update context
        self.context.add_translation(
            source="[audio]",
            target=collected_text,
            speaker_id="default"
        )


class SpeculativeTranslator:
    """Speculative translation using draft + verify approach."""

    def __init__(
        self,
        draft_model_path: str = "zenlm/zen-nano",
        target_model_path: str = "zenlm/zen-omni",
        num_speculative_tokens: int = 4
    ):
        self.draft_translator = ZenOmniTranslator(draft_model_path)
        self.target_translator = ZenOmniTranslator(target_model_path)
        self.num_speculative_tokens = num_speculative_tokens

    def load(self):
        """Load both models."""
        self.draft_translator.load()
        self.target_translator.load()

    async def translate_speculative(
        self,
        audio: np.ndarray,
        source_lang: str = "en",
        target_lang: str = "es"
    ) -> TranslationResult:
        """Translate using speculative decoding."""
        # Draft translation (fast)
        draft_result = self.draft_translator.translate(
            audio, source_lang, target_lang
        )

        # For now, just use draft result
        # Full speculative decoding would verify with target model
        return draft_result


class TranslationStage:
    """Complete translation stage for the pipeline."""

    def __init__(
        self,
        model_path: str = "zenlm/zen-omni",
        source_lang: str = "en",
        target_lang: str = "es",
        use_speculative: bool = False
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.use_speculative = use_speculative

        if use_speculative:
            self.translator = SpeculativeTranslator(
                draft_model_path="zenlm/zen-nano",
                target_model_path=model_path
            )
        else:
            self.translator = ZenOmniTranslator(model_path)

    def load(self):
        """Load translator model(s)."""
        self.translator.load()

    async def translate(self, audio: np.ndarray) -> TranslationResult:
        """Translate audio segment."""
        if self.use_speculative:
            return await self.translator.translate_speculative(
                audio,
                self.source_lang,
                self.target_lang
            )
        else:
            return self.translator.translate(
                audio,
                self.source_lang,
                self.target_lang
            )

    async def translate_streaming(
        self,
        audio: np.ndarray
    ) -> AsyncIterator[TranslationChunk]:
        """Stream translation chunks."""
        if not self.use_speculative:
            async for chunk in self.translator.translate_stream(
                audio,
                self.source_lang,
                self.target_lang
            ):
                yield chunk
