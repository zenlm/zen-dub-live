---
license: apache-2.0
tags:
  - speech-to-speech
  - translation
  - lip-sync
  - dubbing
  - real-time
  - streaming
  - zen-omni
  - zen-dub
  - hanzo-ai
language:
  - en
  - es
  - multilingual
library_name: transformers
pipeline_tag: text-to-speech
---

# Zen-Dub-Live

**Real-Time Speech-to-Speech Translation and Lip-Synchronized Video Dubbing**

Zen-Dub-Live is a streaming pipeline that combines speech-to-speech translation with neural lip-sync rendering. The system achieves glass-to-glass latency of **2.5-3.5 seconds** while maintaining broadcast-quality audio and video output.

## Key Features

- **Ultra-Low Latency**: 2.5-3.5 second end-to-end latency
- **Broadcast Quality**: Production-ready audio and video output
- **Real-Time Lip-Sync**: Neural VAE-based mouth manipulation
- **Streaming Architecture**: Built for live broadcast workflows
- **Anchor Voices**: Consistent voice identity across translations

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Zen-Dub-Live Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Stage 1: Capture     →  Stage 2: Translate  →  Stage 3: Synthesize    │
│  (Audio + Video)         (Zen Omni S2S)         (Anchor Voice + TTS)   │
│                                                                         │
│                       →  Stage 4: Render                                │
│                          (Zen Dub Lip-Sync)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Model | Purpose |
|-----------|-------|---------|
| Translation | [zen-omni](https://huggingface.co/zenlm/zen-omni) | Speech-to-speech translation |
| Lip-Sync | [zen-dub](https://huggingface.co/zenlm/zen-dub) | Neural lip manipulation |
| VAD | Silero VAD | Voice activity detection |

## Installation

```bash
pip install zen-dub-live
```

Or with all dependencies:

```bash
pip install "zen-dub-live[all]"
```

## Quick Start

### Live Dubbing

```bash
# Dub webcam to RTMP stream
zen-dub-live live --source en --target es --output rtmp://localhost/live/stream
```

### File Dubbing

```bash
# Dub a video file
zen-dub-live dub video.mp4 --source en --target es -o dubbed.mp4
```

### Python API

```python
import asyncio
from zen_dub_live import ZenDubLivePipeline, PipelineConfig

config = PipelineConfig(
    source_language="en",
    target_language="es",
)

pipeline = ZenDubLivePipeline(config)

async def main():
    async for segment in pipeline.run():
        print(f"Latency: {segment.latency:.2f}s")

asyncio.run(main())
```

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| E2E Latency | < 3.5s | 2.4s |
| Video FPS | 30+ | 32 |
| Audio Quality | MOS 4.0+ | 4.1 |
| Lip-Sync (LSE-D) | < 8.0 | 7.8 |

## Hardware Requirements

**Minimum** (1080p @ 30fps):
- NVIDIA RTX 4090 or A100 40GB
- 32GB RAM
- 100 Mbps network

**Recommended** (4K @ 60fps):
- 2x NVIDIA A100 80GB
- 64GB RAM
- 1 Gbps network

## Related Models

- [zen-omni](https://huggingface.co/zenlm/zen-omni) - Multimodal translation model
- [zen-dub](https://huggingface.co/zenlm/zen-dub) - Neural lip-sync model
- [zen-nano](https://huggingface.co/zenlm/zen-nano) - Edge inference model

## Links

- **GitHub**: https://github.com/zenlm/zen-dub-live
- **Documentation**: https://zenlm.org/docs/zen-dub-live
- **Paper**: [Technical Whitepaper](https://github.com/zenlm/zen-dub-live/blob/main/paper/zen_dub_live_whitepaper.md)

## License

Apache 2.0

## Citation

```bibtex
@software{zen_dub_live,
  title = {Zen-Dub-Live: Real-Time Speech-to-Speech Translation and Lip-Synchronized Video Dubbing},
  author = {Hanzo AI},
  year = {2025},
  url = {https://github.com/zenlm/zen-dub-live}
}
```

---

**Hanzo AI** | https://hanzo.ai | Techstars '17
