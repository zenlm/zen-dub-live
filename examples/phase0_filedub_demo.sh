#!/usr/bin/env bash
#
# Phase-0 end-to-end (file-based, non-realtime) video-dub demo.
#
# Wires the REAL components:
#   translation  -> hanzo engine OpenAI API (/v1/chat/completions)
#   synthesis    -> hanzo zen3-tts (/v1/audio/speech)        [needs gated codec token]
#   lip-sync     -> zen-dub / MuseTalk (external process)    [needs model weights + GPU]
#   mux          -> ffmpeg
#
# Stages that cannot reach their model on this box fall back gracefully and are
# reported as "skip" so the result is honest about what actually ran.
#
# Usage:
#   examples/phase0_filedub_demo.sh <input.mp4> [output.mp4]
#
# Environment knobs:
#   HANZO_BIN        path to a built `hanzo` engine binary (CPU build recommended here)
#   LLM_MODEL_DIR    local text/LLM model dir to serve for translation
#   TTS_MODEL_DIR    local zen3-tts model dir to serve for synthesis (gated codec)
#   ZEN_DUB_REPO     path to the zen-dub / MuseTalk checkout (default /home/z/work/zen/gh/zen-dub)
#   FFMPEG_BIN/FFPROBE_BIN  explicit ffmpeg paths if not on PATH
#   TRANSCRIPT       source-language transcript (translate via the text LLM)
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT="${1:?usage: phase0_filedub_demo.sh <input.mp4> [output.mp4]}"
OUTPUT="${2:-${INPUT%.*}_dubbed.mp4}"

HANZO_BIN="${HANZO_BIN:-/home/z/work/hanzo-gdn/engine/target-wsl/release/hanzo}"
LLM_MODEL_DIR="${LLM_MODEL_DIR:-/home/z/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
TTS_MODEL_DIR="${TTS_MODEL_DIR:-/home/z/work/zen/hf/zen-3-tts-0.6B}"
SRC="${SRC:-en}"
TGT="${TGT:-es}"
TRANSCRIPT="${TRANSCRIPT:-Good morning everyone. Welcome to our demonstration of real time video dubbing.}"

LLM_PORT=8124
TTS_PORT=8125

echo "== phase0 file dub =="
echo "  input : $INPUT"
echo "  output: $OUTPUT"
echo "  $SRC -> $TGT"

cleanup() {
  [ -n "${LLM_PID:-}" ] && kill "$LLM_PID" 2>/dev/null || true
  [ -n "${TTS_PID:-}" ] && kill "$TTS_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- start translation LLM (CPU, OpenAI server) ---------------------------- #
echo "-- starting hanzo LLM server on :$LLM_PORT (CPU) ..."
"$HANZO_BIN" serve -p "$LLM_PORT" text -m "$LLM_MODEL_DIR" --cpu > /tmp/phase0_llm.log 2>&1 &
LLM_PID=$!
for i in $(seq 1 60); do
  if (exec 3<>/dev/tcp/127.0.0.1/$LLM_PORT) 2>/dev/null; then echo "   LLM up"; break; fi
  sleep 1
done

# --- (optional) start zen3-tts server -------------------------------------- #
# This requires an HF token for the gated codec `hanzoai/dac_44khz`. If it
# fails to load, the demo falls back to a placeholder voice automatically.
TTS_URL=""
if [ -d "$TTS_MODEL_DIR" ]; then
  echo "-- starting hanzo zen3-tts server on :$TTS_PORT (CPU) ..."
  "$HANZO_BIN" serve -p "$TTS_PORT" speech -m "$TTS_MODEL_DIR" --cpu > /tmp/phase0_tts.log 2>&1 &
  TTS_PID=$!
  for i in $(seq 1 30); do
    if (exec 3<>/dev/tcp/127.0.0.1/$TTS_PORT) 2>/dev/null; then TTS_URL="http://localhost:$TTS_PORT"; echo "   TTS up"; break; fi
    sleep 1
  done
  [ -z "$TTS_URL" ] && echo "   TTS unavailable (likely gated codec); using placeholder voice"
fi

# --- run the e2e dub ------------------------------------------------------- #
cd "$ROOT"
TTS_ARG=()
[ -n "$TTS_URL" ] && TTS_ARG=(--tts-url "$TTS_URL")

PYTHONPATH="$ROOT/src" python3 -m zen_dub_live.cli dub-file \
  "$INPUT" -o "$OUTPUT" \
  --source "$SRC" --target "$TGT" \
  --transcript "$TRANSCRIPT" \
  --llm-url "http://localhost:$LLM_PORT" \
  "${TTS_ARG[@]}"

echo
echo "== done: $OUTPUT =="
