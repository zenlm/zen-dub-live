"""
Hanzo Engine client (real wiring).

This module replaces the model-internal stubs with concrete calls to a running
``hanzo`` engine (https OpenAI-compatible API) or, as a fallback, the ``hanzo``
CLI in one-shot mode.

The whole module is intentionally dependency-light: it uses only the Python
standard library (``urllib``, ``json``, ``subprocess``, ``wave``) so it runs on
a bare interpreter with just numpy. No ``torch``/``transformers``/``openai``
import is required to drive the engine -- the engine does the heavy lifting in a
separate process, exactly like the production zen-live-dub topology.

Endpoints used (registered by hanzo-server-core):
  POST /v1/chat/completions   -> translation / ASR (text or multimodal models)
  POST /v1/audio/speech       -> zen3-tts speech synthesis (OpenAI audio API)
  GET  /v1/models             -> health / model id discovery

Start an engine like this (CPU example, fully offline once weights are local):

  hanzo serve -p 8124 text -m <local-qwen3-or-zen-text-model> --cpu
  hanzo serve -p 8125 speech -m <local-zen3-tts-model> --cpu   # needs HF token for the gated codec

Or run one-shot via the CLI (no server):

  hanzo run -m <model> text     # interactive
  hanzo run -m <tts-model> speech
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# A built CPU-only hanzo binary on this box (WSL build). Overridable via env.
DEFAULT_HANZO_BIN = os.environ.get(
    "HANZO_BIN", "/home/z/work/hanzo-gdn/engine/target-wsl/release/hanzo"
)


@dataclass
class EngineConfig:
    """How to reach a hanzo engine for one modality (LLM or TTS)."""

    # If set, talk to a running server over HTTP (preferred).
    base_url: Optional[str] = None          # e.g. "http://localhost:8124"
    model: str = "default"                  # OpenAI "model" field; "default" works for single-model servers
    api_key: str = "EMPTY"
    timeout_s: float = 600.0

    # CLI fallback (used only when base_url is None / server unreachable).
    hanzo_bin: str = DEFAULT_HANZO_BIN
    model_path: Optional[str] = None        # local model dir for `hanzo run ...`

    def is_http(self) -> bool:
        return bool(self.base_url)


# --------------------------------------------------------------------------- #
# Low-level HTTP helper (stdlib only)
# --------------------------------------------------------------------------- #

def _post_json(url: str, payload: Dict[str, Any], api_key: str, timeout_s: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _post_raw(url: str, payload: Dict[str, Any], api_key: str, timeout_s: float) -> bytes:
    """POST JSON, return the raw response bytes (for binary audio)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def engine_is_up(base_url: str, timeout_s: float = 5.0) -> bool:
    """Return True if /v1/models responds."""
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Chat completion (translation / ASR)
# --------------------------------------------------------------------------- #

def _extract_chat_text(resp: Dict[str, Any]) -> str:
    """Pull assistant text out of a (possibly partial/errored) chat response."""
    # hanzo returns {"message": "...", "partial_response": {...}} on mid-gen error.
    container = resp
    if "choices" not in container and "partial_response" in container:
        container = container["partial_response"]
    choices = container.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return msg.get("content") or ""


def strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks and tidy whitespace."""
    out = text
    while "<think>" in out and "</think>" in out:
        pre, _, rest = out.partition("<think>")
        _, _, post = rest.partition("</think>")
        out = pre + post
    return out.strip()


def chat_complete(
    cfg: EngineConfig,
    messages: List[Dict[str, Any]],
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    """
    Run a chat completion against a hanzo engine.

    Returns the assistant text (thinking blocks stripped). Raises on hard
    failure with no usable partial output.
    """
    if cfg.is_http():
        url = cfg.base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = _post_json(url, payload, cfg.api_key, cfg.timeout_s)
        except urllib.error.HTTPError as e:
            # hanzo may still return JSON with a partial_response on 4xx/5xx.
            try:
                resp = json.loads(e.read().decode("utf-8"))
            except Exception:
                raise
        text = strip_think(_extract_chat_text(resp))
        if not text:
            raise RuntimeError(f"engine returned empty completion: {resp!r}")
        return text

    # CLI fallback: hanzo run ... text  (one-shot via stdin)
    return _chat_complete_cli(cfg, messages, max_tokens, temperature)


def _chat_complete_cli(cfg, messages, max_tokens, temperature) -> str:
    if not cfg.model_path:
        raise RuntimeError("CLI fallback needs cfg.model_path")
    # Flatten messages into a single prompt for one-shot mode.
    prompt_lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        prompt_lines.append(f"{role}: {content}")
    prompt = "\n".join(prompt_lines) + "\nassistant:"
    cmd = [cfg.hanzo_bin, "run", "-m", cfg.model_path, "text"]
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=cfg.timeout_s
    )
    if proc.returncode != 0:
        raise RuntimeError(f"hanzo run failed: {proc.stderr[-500:]}")
    return strip_think(proc.stdout.strip())


# --------------------------------------------------------------------------- #
# Speech synthesis (zen3-tts)
# --------------------------------------------------------------------------- #

@dataclass
class SpeechResult:
    audio_path: str           # path to a 16-bit PCM .wav written on disk
    sample_rate: int
    backend: str              # "engine-http" | "engine-cli" | "fallback-formant"
    note: str = ""


def synthesize_speech(
    cfg: EngineConfig,
    text: str,
    out_wav: str,
    voice: str = "cherry",
    sample_rate: int = 24000,
) -> SpeechResult:
    """
    Synthesize ``text`` to a wav file using the hanzo zen3-tts engine.

    Tries, in order:
      1. POST /v1/audio/speech  (OpenAI audio API)  -> writes returned bytes
      2. hanzo run speech  (CLI, if model_path given)
    Raises if neither is available; callers may catch and use a fallback.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)

    if cfg.is_http():
        url = cfg.base_url.rstrip("/") + "/v1/audio/speech"
        payload = {
            "model": cfg.model,
            "input": text,
            "voice": voice,
            "response_format": "wav",
        }
        raw = _post_raw(url, payload, cfg.api_key, cfg.timeout_s)
        with open(out_wav, "wb") as f:
            f.write(raw)
        return SpeechResult(out_wav, sample_rate, "engine-http")

    if cfg.model_path:
        # hanzo run speech writes a wav; the exact flag set is model-dependent.
        cmd = [cfg.hanzo_bin, "run", "-m", cfg.model_path, "speech"]
        proc = subprocess.run(
            cmd, input=text, capture_output=True, text=True, timeout=cfg.timeout_s
        )
        if proc.returncode != 0:
            raise RuntimeError(f"hanzo run speech failed: {proc.stderr[-500:]}")
        # The CLI prints the output path; if it wrote out_wav, use it.
        if os.path.exists(out_wav):
            return SpeechResult(out_wav, sample_rate, "engine-cli")
        raise RuntimeError("hanzo run speech produced no wav at expected path")

    raise RuntimeError("no TTS engine configured (set base_url or model_path)")
