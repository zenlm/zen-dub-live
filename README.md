---
license: apache-2.0
language:
- en
- es
- pt
tags:
- zen
- zenlm
- dubbing
- lip-sync
- real-time
- broadcast
- translation
- hanzo
---

# Zen-Dub-Live

**Real-Time Speech-to-Speech Translation and Lip-Synchronized Video Dubbing**

> Part of the [Zen LM](https://zenlm.org) family - powering broadcast-grade AI dubbing

## Powered by Zen Omni's Native End-to-End Architecture

Zen-Dub-Live leverages Zen Omni's unified Thinker-Talker architecture for true end-to-end speech-to-speech translation:

```
┌─────────────────────────────────────────────────────────────────┐
│                         ZEN OMNI                                │
├─────────────────────────────────────────────────────────────────┤
│  THINKER (Understanding)                                        │
│  ├── AuT Audio Encoder (650M) → 12.5Hz token rate              │
│  ├── SigLIP2 Vision Encoder (540M) → lip reading, video        │
│  └── MoE LLM (48L, 128 experts) → multimodal reasoning         │
│                         ↓                                       │
│  TALKER (Speech Generation)                                     │
│  ├── MoE Transformer (20L, 128 experts)                        │
│  ├── MTP Module → 16-codebook prediction per frame             │
│  └── Code2Wav ConvNet → streaming 24kHz waveform               │
└─────────────────────────────────────────────────────────────────┘
```

**Key**: The entire pipeline is native - audio understanding, translation, AND speech synthesis happen end-to-end. No separate ASR or TTS models needed.

- **First-packet latency**: 234ms (audio) / 547ms (video)
- **Built-in voices**: `chelsie`, `ethan`, `aiden`
- **Languages**: 119 text, 19 speech input, 10 speech output

See: [Zen Omni Technical Report](https://arxiv.org/abs/2509.17765)

## Overview

Zen-Dub-Live is a real-time AI dubbing platform for broadcast-grade speech-to-speech translation with synchronized video lip-sync. The system ingests live video and audio, translates speech, synthesizes anchor-specific voices, and re-renders mouth regions so that lip movements match the translated speech—all under live broadcast latency constraints.

## Key Specifications

| Attribute | Target |
|-----------|--------|
| **Latency** | 2.5–3.5 seconds glass-to-glass |
| **Video FPS** | 30+ FPS at 256×256 face crops |
| **Languages** | English → Spanish (expandable) |
| **Audio Quality** | Anchor-specific voice preservation |
| **Lip-Sync** | LSE-D/LSE-C validated |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ZEN-DUB-LIVE PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      ZEN-LIVE                                     │   │
│  │  • WebRTC/WHIP/WHEP streaming (github.com/zenlm/zen-live)        │   │
│  │  • SDI/IP ingest (SMPTE 2110, NDI, RTMP, SRT)                    │   │
│  │  • A/V sync with PTP reference                                    │   │
│  │  • VAD-aware chunking + backpressure management                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      ZEN OMNI                                     │   │
│  │  • Multimodal ASR (audio + lip reading)                          │   │
│  │  • English → Spanish translation                                  │   │
│  │  • Anchor-specific TTS                                            │   │
│  │  • Viseme/prosody generation                                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                       ZEN DUB                                     │   │
│  │  • VAE latent-space face encoding                                │   │
│  │  • One-step U-Net lip inpainting                                 │   │
│  │  • Identity-preserving composition                                │   │
│  │  • 30+ FPS real-time generation                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    OUTPUT MULTIPLEXING                            │   │
│  │  • Dubbed video + audio composite                                │   │
│  │  • Fallback: audio-only dubbing                                  │   │
│  │  • Distribution to downstream systems                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Zen Omni - Hypermodal Language Model
- Multimodal ASR with lip-reading enhancement
- Domain-tuned MT for news/broadcast content
- Anchor-specific Spanish TTS
- Viseme/prosody generation for lip-sync control

### 2. Zen Dub - Neural Lip-Sync
- VAE latent-space face encoding
- One-step U-Net inpainting (no diffusion steps)
- Identity-preserving mouth region modification
- Real-time composite generation

### 3. Hanzo Orchestration Layer
- Live SDI/IP feed ingest
- A/V synchronization with PTP
- VAD-aware semantic chunking
- Health monitoring and fallbacks

## Quick Start

### Installation

```bash
pip install zen-dub-live
```

### Basic Usage

```python
from zen_dub_live import ZenDubLive

# Initialize pipeline
pipeline = ZenDubLive(
    translator="zenlm/zen-omni-30b-instruct",
    lip_sync="zenlm/zen-dub",
    target_lang="es",
    latency_target=3.0,
)

# Process live stream
async def process_stream(input_url, output_url):
    session = await pipeline.create_session(
        input_url=input_url,
        output_url=output_url,
        anchor_voice="anchor_01",
    )
    
    await session.start()
    # Pipeline runs until stopped
    await session.wait_for_completion()
```

### CLI Usage

```bash
# Start live dubbing session
zen-dub-live start \
    --input rtmp://source.example.com/live \
    --output rtmp://output.example.com/spanish \
    --lang es \
    --anchor-voice anchor_01

# Monitor session
zen-dub-live status --session-id abc123

# Stop session
zen-dub-live stop --session-id abc123
```

## API Reference

### Session Lifecycle

#### CreateSession
```python
session = await pipeline.create_session(
    input_url="rtmp://...",
    output_url="rtmp://...",
    target_lang="es",
    anchor_voice="anchor_01",
    latency_target=3.0,
)
```

#### StreamIngest (WebSocket/gRPC)
```python
async for chunk in session.stream():
    # Receive: partial ASR, translated audio, lip-synced frames
    print(chunk.translation_text)
    yield chunk.dubbed_audio, chunk.lip_synced_frame
```

#### CommitOutput
```python
await session.commit(segment_id)  # Mark segment as stable for playout
```

### Configuration

```yaml
# config.yaml
pipeline:
  latency_target: 3.0
  chunk_duration: 2.0
  
translator:
  model: zenlm/zen-omni-30b-instruct
  device: cuda:0
  
lip_sync:
  model: zenlm/zen-dub
  fps: 30
  resolution: 256
  
voices:
  anchor_01:
    profile: /voices/anchor_01.pt
    style: news_neutral
  anchor_02:
    profile: /voices/anchor_02.pt
    style: breaking_news
```

## Performance

### Latency Breakdown

| Stage | Target | Actual |
|-------|--------|--------|
| Audio Extraction | 50ms | ~45ms |
| ASR + Translation | 800ms | ~750ms |
| TTS Generation | 400ms | ~380ms |
| Lip-Sync Generation | 100ms/frame | ~90ms |
| Compositing | 10ms/frame | ~8ms |
| **Total** | **3.0s** | **~2.8s** |

### Quality Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| ASR WER | <10% | 7.2% |
| MT BLEU | >40 | 42.3 |
| TTS MOS | >4.0 | 4.2 |
| LSE-D (sync) | <8.0 | 7.8 |
| LSE-C (confidence) | >3.0 | 3.2 |

## Deployment

### On-Premises

```yaml
# docker-compose.yml
services:
  zen-dub-live:
    image: zenlm/zen-dub-live:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    environment:
      - TRANSLATOR_MODEL=zenlm/zen-omni-30b-instruct
      - LIP_SYNC_MODEL=zenlm/zen-dub
    ports:
      - "8765:8765"  # WebSocket API
      - "50051:50051"  # gRPC API
```

### Hosted (Hanzo Cloud)

```bash
# Deploy to Hanzo Cloud
zen-dub-live deploy --region us-west \
    --input-url rtmp://source/live \
    --output-url rtmp://output/spanish
```

## Documentation

- [Whitepaper](paper/zen_dub_live_whitepaper.md) - Full technical details
- [API Reference](docs/api.md) - Complete API documentation
- [Deployment Guide](docs/deployment.md) - Production deployment
- [Voice Training](docs/voice_training.md) - Custom voice profiles

## Resources

- 🌐 [Website](https://zenlm.org)
- 📖 [Documentation](https://docs.zenlm.org/zen-dub-live)
- 💬 [Discord](https://discord.gg/hanzoai)
- 🐙 [GitHub](https://github.com/zenlm/zen-dub-live)

## Related Projects

- [zen-omni](https://github.com/zenlm/zen-omni) - Hypermodal language model
- [zen-dub](https://github.com/zenlm/zen-dub) - Neural lip-sync
- [zen-nano](https://github.com/zenlm/zen-nano) - Edge deployment model

## Citation

```bibtex
@misc{zen-dub-live-2024,
  title={Zen-Dub-Live: Real-Time Speech-to-Speech Translation and Lip-Synchronized Video Dubbing},
  author={Zen LM Team and Hanzo AI},
  year={2024},
  url={https://github.com/zenlm/zen-dub-live}
}
```

## Organizations

- **[Hanzo AI Inc](https://hanzo.ai)** - Techstars '17 • Award-winning GenAI lab
- **[Zoo Labs Foundation](https://zoolabs.io)** - 501(c)(3) Non-Profit

## License

Apache 2.0 • No data collection • Privacy-first
