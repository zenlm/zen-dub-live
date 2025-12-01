"""
Zen-Dub-Live: Real-Time Speech-to-Speech Translation and Lip-Synchronized Video Dubbing

This package provides a streaming pipeline for live dubbing with:
- VAD-aware speech segmentation
- Streaming translation via Zen Omni
- Anchor voice synthesis
- Real-time lip-sync rendering via Zen Dub
"""

__version__ = "0.1.0"
__author__ = "Hanzo AI"
__license__ = "Apache-2.0"

from .pipeline import ZenDubLivePipeline, PipelineConfig
from .capture import AudioCapture, VideoCapture, StreamSynchronizer
from .translation import ZenOmniTranslator, TranslationContext
from .synthesis import AnchorVoice, ProsodyTransfer, VisemeGenerator
from .render import ZenDubRenderer, OneStepInpainter, TemporalSmoother
from .orchestration import ZenLiveOrchestrator, HanzoOrchestrator, BroadcastOutput

__all__ = [
    # Pipeline
    "ZenDubLivePipeline",
    "PipelineConfig",
    # Capture
    "AudioCapture",
    "VideoCapture",
    "StreamSynchronizer",
    # Translation
    "ZenOmniTranslator",
    "TranslationContext",
    # Synthesis
    "AnchorVoice",
    "ProsodyTransfer",
    "VisemeGenerator",
    # Render
    "ZenDubRenderer",
    "OneStepInpainter",
    "TemporalSmoother",
    # Orchestration (zen-live integration)
    "ZenLiveOrchestrator",
    "HanzoOrchestrator",  # Alias for backwards compat
    "BroadcastOutput",
]
