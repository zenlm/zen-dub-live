"""
File-based (non-realtime) end-to-end dubbing driver.

This is the concrete pipeline that wires the REAL components together to dub a
short clip from disk:

    input.mp4
       |
       |  (1) extract source audio  ............................  ffmpeg
       v
    source.wav
       |
       |  (2) ASR + translate  .................................  hanzo engine
       |        - transcript provided  -> translate_text (LLM /v1/chat)
       |        - audio-only model     -> asr_translate  (omni/VLM /v1/chat)
       v
    target_text  (e.g. Spanish)
       |
       |  (3) synthesize speech  ...............................  hanzo zen3-tts
       |        - POST /v1/audio/speech (or `hanzo run speech`)
       |        - placeholder voice fallback if codec unavailable
       v
    dub.wav
       |
       |  (4) repaint mouth  ...................................  zen-dub / MuseTalk
       |        - external `python -m scripts.inference`
       |        - identity (copy frames) fallback if no weights/GPU
       v
    dubbed_video.mp4 (no audio)
       |
       |  (5) mux dubbed video + dub.wav  ......................  ffmpeg
       v
    OUTPUT.mp4

Each stage reports whether it RAN for real or used a fallback, so the result is
honest about what was actually exercised.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .engine_client import EngineConfig, engine_is_up
from .translation import HanzoEngineTranslator
from .synthesis import HanzoTTSSynthesizer
from .render import MuseTalkRenderer


# --------------------------------------------------------------------------- #
# ffmpeg discovery
# --------------------------------------------------------------------------- #

def _find_tool(name: str, env_var: str) -> Optional[str]:
    cand = os.environ.get(env_var)
    if cand and os.path.exists(cand):
        return cand
    found = shutil.which(name)
    if found:
        return found
    # Local wrapper produced during bring-up on this box.
    local = f"/tmp/ffbin/{name}"
    if os.path.exists(local):
        return local
    return None


FFMPEG = _find_tool("ffmpeg", "FFMPEG_BIN")
FFPROBE = _find_tool("ffprobe", "FFPROBE_BIN")


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stderr[-800:]}")


# --------------------------------------------------------------------------- #
# Stage report
# --------------------------------------------------------------------------- #

@dataclass
class StageReport:
    name: str
    ran: bool
    backend: str
    detail: str = ""


@dataclass
class DubReport:
    input: str
    output: str
    target_text: str = ""
    stages: List[StageReport] = field(default_factory=list)

    def add(self, name, ran, backend, detail=""):
        self.stages.append(StageReport(name, ran, backend, detail))

    def to_dict(self) -> Dict:
        return {
            "input": self.input,
            "output": self.output,
            "target_text": self.target_text,
            "stages": [vars(s) for s in self.stages],
        }


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #

def probe_duration(path: str) -> float:
    if not FFPROBE:
        return 0.0
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def has_audio_stream(video_path: str) -> bool:
    if not FFPROBE:
        return False
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "default=nw=1:nk=1", video_path],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


def extract_audio(video_path: str, out_wav: str, sample_rate: int = 16000) -> bool:
    """Extract mono audio to a wav. Returns False (no-op) if the clip is silent."""
    if not FFMPEG:
        raise RuntimeError("ffmpeg not found (set FFMPEG_BIN)")
    if not has_audio_stream(video_path):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    _run([FFMPEG, "-y", "-v", "error", "-i", video_path,
          "-vn", "-ac", "1", "-ar", str(sample_rate), out_wav])
    return True


def strip_audio(video_path: str, out_video: str) -> None:
    """Produce a video with no audio track (so we can mux the dub cleanly)."""
    if not FFMPEG:
        raise RuntimeError("ffmpeg not found")
    os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
    _run([FFMPEG, "-y", "-v", "error", "-i", video_path,
          "-an", "-c:v", "copy", out_video])


def mux(video_path: str, audio_wav: str, out_path: str) -> None:
    """Mux a (silent) video with an audio wav, trimming to the shorter stream."""
    if not FFMPEG:
        raise RuntimeError("ffmpeg not found")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    _run([FFMPEG, "-y", "-v", "error",
          "-i", video_path, "-i", audio_wav,
          "-map", "0:v:0", "-map", "1:a:0",
          "-c:v", "copy", "-c:a", "aac", "-shortest", out_path])


# --------------------------------------------------------------------------- #
# The end-to-end driver
# --------------------------------------------------------------------------- #

def dub_file(
    input_video: str,
    output_video: str,
    *,
    source_lang: str = "en",
    target_lang: str = "es",
    transcript: Optional[str] = None,
    llm_url: Optional[str] = None,
    llm_model: str = "default",
    tts_url: Optional[str] = None,
    tts_model: str = "default",
    tts_model_path: Optional[str] = None,
    use_musetalk: bool = True,
    musetalk_repo: Optional[str] = None,
    workdir: Optional[str] = None,
    allow_tts_fallback: bool = True,
) -> DubReport:
    """
    Dub ``input_video`` -> ``output_video``. Returns a DubReport detailing which
    stages ran for real vs fell back.

    Translation source:
      * pass ``transcript`` to translate known text via the LLM (fully real), or
      * omit it to attempt direct ASR+translate from the clip's audio using an
        omni/VLM model at ``llm_url`` (real only if that model accepts audio).
    """
    report = DubReport(input=input_video, output=output_video)
    work = workdir or os.path.join(os.path.dirname(os.path.abspath(output_video)) or ".",
                                   "_zendub_work")
    os.makedirs(work, exist_ok=True)

    src_wav = os.path.join(work, "source16k.wav")
    dub_wav = os.path.join(work, "dub.wav")
    silent_mp4 = os.path.join(work, "silent.mp4")
    lips_mp4 = os.path.join(work, "lipsync.mp4")

    # ---- (1) extract source audio --------------------------------------- #
    have_src_audio = extract_audio(input_video, src_wav, sample_rate=16000)
    if have_src_audio:
        report.add("extract_audio", ran=True, backend="ffmpeg",
                   detail=f"{src_wav} ({probe_duration(src_wav):.1f}s)")
    else:
        report.add("extract_audio", ran=False, backend="ffmpeg",
                   detail="clip has no audio track (ASR path unavailable; "
                          "translation must use --transcript)")

    # ---- (2) ASR + translate via hanzo engine --------------------------- #
    llm_cfg = EngineConfig(base_url=llm_url, model=llm_model) if llm_url else None
    if llm_cfg is None or not engine_is_up(llm_cfg.base_url):
        if transcript is None:
            report.add("translate", ran=False, backend="none",
                       detail="no LLM engine reachable and no transcript provided")
            target_text = ""
        else:
            report.add("translate", ran=False, backend="none",
                       detail="no LLM engine reachable; passing transcript through untranslated")
            target_text = transcript
    else:
        translator = HanzoEngineTranslator(llm_cfg, source_lang, target_lang)
        if transcript is not None:
            res = translator.translate_text(transcript)
            report.add("translate", ran=True, backend="hanzo-llm",
                       detail=f"text->text via /v1/chat ({res.duration:.1f}s)")
            target_text = res.target_text
        elif have_src_audio:
            res = translator.asr_translate(src_wav)
            report.add("translate", ran=True, backend="hanzo-omni",
                       detail=f"audio->text via /v1/chat ({res.duration:.1f}s)")
            target_text = res.target_text
        else:
            report.add("translate", ran=False, backend="none",
                       detail="no transcript and no source audio to ASR")
            target_text = ""
    report.target_text = target_text

    # ---- (3) synthesize speech ------------------------------------------ #
    tts_cfg = None
    if tts_url:
        tts_cfg = EngineConfig(base_url=tts_url, model=tts_model)
    elif tts_model_path:
        tts_cfg = EngineConfig(model_path=tts_model_path)
    synth = HanzoTTSSynthesizer(tts_cfg, voice="cherry", sample_rate=24000,
                                allow_fallback=allow_tts_fallback)
    speech = synth.synthesize_to_file(target_text or " ", dub_wav)
    report.add("tts", ran=(speech.backend.startswith("engine")),
               backend=speech.backend,
               detail=speech.note or f"{speech.audio_path} @ {speech.sample_rate}Hz")

    # ---- (4) lip-sync repaint via MuseTalk ------------------------------ #
    did_lipsync = False
    if use_musetalk:
        try:
            renderer = MuseTalkRenderer(
                repo_dir=musetalk_repo or os.environ.get(
                    "ZEN_DUB_REPO", "/home/z/work/zen/gh/zen-dub")
            )
            mt = renderer.render_clip(input_video, dub_wav, lips_mp4)
            report.add("lipsync", ran=mt.ran, backend=mt.backend, detail=lips_mp4)
            did_lipsync = mt.ran
        except Exception as e:
            report.add("lipsync", ran=False, backend="skipped",
                       detail=f"MuseTalk unavailable ({type(e).__name__}: {e}); "
                              f"using source frames unchanged")
    else:
        report.add("lipsync", ran=False, backend="disabled", detail="use_musetalk=False")

    # ---- (5) mux -------------------------------------------------------- #
    if did_lipsync and os.path.exists(lips_mp4):
        # MuseTalk output already has the dub audio muxed in; just copy it out.
        shutil.copyfile(lips_mp4, output_video)
        report.add("mux", ran=True, backend="musetalk-internal", detail=output_video)
    else:
        # No lip-sync: strip original audio, mux the new dub track onto the
        # original video so the result is the original frames + translated speech.
        strip_audio(input_video, silent_mp4)
        mux(silent_mp4, dub_wav, output_video)
        report.add("mux", ran=True, backend="ffmpeg", detail=output_video)

    return report


def format_report(report: DubReport) -> str:
    lines = []
    lines.append(f"input : {report.input}")
    lines.append(f"output: {report.output}")
    lines.append(f"target_text: {report.target_text!r}")
    lines.append("stages:")
    for s in report.stages:
        mark = "RAN " if s.ran else "skip"
        lines.append(f"  [{mark}] {s.name:14s} backend={s.backend:18s} {s.detail}")
    return "\n".join(lines)
