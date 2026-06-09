"""
Smoke tests for the phase-0 file-based dub wiring.

These do NOT require a GPU, the hanzo engine, or MuseTalk weights. They stand up
tiny stdlib HTTP mocks of the engine's OpenAI endpoints and verify that:

  * engine_client.chat_complete does a real /v1/chat/completions round-trip
  * engine_client.synthesize_speech does a real /v1/audio/speech round-trip
  * synthesis fallback produces real (non-silent) audio when no engine is set

ffmpeg-dependent stages (extract/mux) are covered by the example script and the
manual e2e run documented in the README; they are skipped here if ffmpeg is
absent.
"""

import io
import json
import math
import struct
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

try:
    import pytest
except ImportError:  # allow running standalone without pytest installed
    class _PytestShim:
        def fixture(self, *a, **k):
            def deco(fn):
                return fn
            return deco
    pytest = _PytestShim()

from zen_dub_live.engine_client import (
    EngineConfig,
    chat_complete,
    synthesize_speech,
    strip_think,
)
from zen_dub_live.synthesis import HanzoTTSSynthesizer, _placeholder_voice


def _wav_bytes(seconds=1.0, sr=24000):
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(b"".join(
        struct.pack("<h", int(10000 * math.sin(2 * math.pi * 200 * i / sr)))
        for i in range(int(sr * seconds))
    ))
    w.close()
    return buf.getvalue()


class _MockEngine(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json({"object": "list", "data": [{"id": "mock"}]})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(ln) or b"{}")
        if self.path.endswith("/chat/completions"):
            # Echo a deterministic "translation".
            self._json({
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "<think>reasoning</think>HOLA MUNDO"
                }}]
            })
        elif self.path.endswith("/audio/speech"):
            wav = _wav_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)
        else:
            self.send_response(404); self.end_headers()

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


@pytest.fixture()
def mock_engine():
    srv = HTTPServer(("127.0.0.1", 0), _MockEngine)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def test_strip_think():
    assert strip_think("<think>x</think> hi") == "hi"
    assert strip_think("plain") == "plain"


def test_chat_complete_roundtrip(mock_engine):
    cfg = EngineConfig(base_url=mock_engine, model="mock")
    out = chat_complete(cfg, [{"role": "user", "content": "hi"}])
    assert out == "HOLA MUNDO"  # think block stripped


def test_synthesize_speech_http(mock_engine, tmp_path):
    cfg = EngineConfig(base_url=mock_engine, model="mock")
    out = tmp_path / "s.wav"
    res = synthesize_speech(cfg, "hola", str(out))
    assert res.backend == "engine-http"
    assert out.exists() and out.stat().st_size > 1000


def test_tts_fallback_is_real_audio(tmp_path):
    synth = HanzoTTSSynthesizer(cfg=None, allow_fallback=True)
    out = tmp_path / "fb.wav"
    res = synth.synthesize_to_file("buenos dias esto es una prueba", str(out))
    assert res.backend == "fallback-formant"
    with wave.open(str(out), "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert np.abs(a).max() > 100  # non-silent


def test_placeholder_voice_length_scales_with_text():
    short = _placeholder_voice("hi")
    long = _placeholder_voice("this is a much much longer sentence to synthesize aloud")
    assert len(long) > len(short)
