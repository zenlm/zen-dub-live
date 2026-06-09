"""
Zen-Dub-Live CLI

Command-line interface for real-time dubbing.
"""

import argparse
import asyncio
import sys
from typing import Optional


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zen-Dub-Live: Real-Time Speech Translation & Lip-Sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live dubbing from webcam to RTMP
  zen-dub-live live --source en --target es --output rtmp://localhost/live/stream

  # Dub a video file
  zen-dub-live dub video.mp4 --source en --target es -o dubbed.mp4

  # Run with custom models
  zen-dub-live live --model zenlm/zen-omni --render-model zenlm/zen-dub

For more information: https://zenlm.org/docs/zen-dub-live
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Live command
    live_parser = subparsers.add_parser("live", help="Start live dubbing")
    live_parser.add_argument(
        "--source", "-s",
        default="en",
        help="Source language code (default: en)"
    )
    live_parser.add_argument(
        "--target", "-t",
        default="es",
        help="Target language code (default: es)"
    )
    live_parser.add_argument(
        "--output", "-o",
        default="rtmp://localhost/live/stream",
        help="Output URL (RTMP, SRT, or file path)"
    )
    live_parser.add_argument(
        "--video-source",
        default="0",
        help="Video source (device index or URL, default: 0)"
    )
    live_parser.add_argument(
        "--audio-source",
        default="default",
        help="Audio source (device name, default: default)"
    )
    live_parser.add_argument(
        "--model",
        default="zenlm/zen-omni",
        help="Translation model (default: zenlm/zen-omni)"
    )
    live_parser.add_argument(
        "--render-model",
        default="zenlm/zen-dub",
        help="Lip-sync render model (default: zenlm/zen-dub)"
    )
    live_parser.add_argument(
        "--anchor-voice",
        help="Anchor voice ID for consistent voice"
    )

    # Dub command
    dub_parser = subparsers.add_parser("dub", help="Dub a video file")
    dub_parser.add_argument("input", help="Input video file")
    dub_parser.add_argument(
        "--output", "-o",
        help="Output video file (default: input_dubbed.mp4)"
    )
    dub_parser.add_argument(
        "--source", "-s",
        default="en",
        help="Source language code (default: en)"
    )
    dub_parser.add_argument(
        "--target", "-t",
        default="es",
        help="Target language code (default: es)"
    )
    dub_parser.add_argument(
        "--model",
        default="zenlm/zen-omni",
        help="Translation model"
    )

    # File-based end-to-end dub (non-realtime) -- the phase0 demo path
    fd = subparsers.add_parser(
        "dub-file",
        help="File-based e2e dub: ASR/translate (hanzo) -> zen3-tts -> MuseTalk -> ffmpeg mux",
    )
    fd.add_argument("input", help="Input video file (short talking-head clip)")
    fd.add_argument("--output", "-o", help="Output dubbed video (default: <input>_dubbed.mp4)")
    fd.add_argument("--source", "-s", default="en", help="Source language code")
    fd.add_argument("--target", "-t", default="es", help="Target language code")
    fd.add_argument("--transcript", help="Known source transcript (translate via text LLM)")
    fd.add_argument("--transcript-file", help="Path to a file containing the source transcript")
    fd.add_argument("--llm-url", help="Hanzo LLM server base URL, e.g. http://localhost:8124")
    fd.add_argument("--llm-model", default="default", help="LLM model id for the server")
    fd.add_argument("--tts-url", help="Hanzo zen3-tts server base URL, e.g. http://localhost:8125")
    fd.add_argument("--tts-model", default="default", help="TTS model id for the server")
    fd.add_argument("--tts-model-path", help="Local zen3-tts model dir for `hanzo run speech` fallback")
    fd.add_argument("--no-musetalk", action="store_true", help="Skip MuseTalk lip-sync")
    fd.add_argument("--musetalk-repo", help="Path to the zen-dub/MuseTalk checkout")
    fd.add_argument("--no-tts-fallback", action="store_true",
                    help="Fail instead of using the placeholder voice when TTS engine is down")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show pipeline status")

    # Voices command
    voices_parser = subparsers.add_parser("voices", help="List available anchor voices")

    args = parser.parse_args()

    if args.command == "live":
        asyncio.run(cmd_live(args))
    elif args.command == "dub":
        asyncio.run(cmd_dub(args))
    elif args.command == "dub-file":
        cmd_dub_file(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "voices":
        cmd_voices(args)
    else:
        parser.print_help()
        sys.exit(1)


async def cmd_live(args):
    """Run live dubbing."""
    from .pipeline import ZenDubLivePipeline, PipelineConfig
    from .orchestration import OutputProtocol

    # Determine output protocol
    output_url = args.output
    if output_url.startswith("rtmp://"):
        protocol = OutputProtocol.RTMP
        output_config = {"url": output_url}
    elif output_url.startswith("srt://"):
        protocol = OutputProtocol.SRT
        port = int(output_url.split(":")[-1])
        output_config = {"port": port}
    else:
        protocol = OutputProtocol.FILE
        output_config = {"path": output_url}

    config = PipelineConfig(
        audio_source=args.audio_source,
        video_source=args.video_source,
        model_path=args.model,
        source_language=args.source,
        target_language=args.target,
        render_model_path=args.render_model,
        output_protocol=protocol,
        output_config=output_config
    )

    if args.anchor_voice:
        config.anchor_voice_id = args.anchor_voice

    pipeline = ZenDubLivePipeline(config)

    print(f"🎬 Zen-Dub-Live")
    print(f"   Source: {args.source} → Target: {args.target}")
    print(f"   Output: {output_url}")
    print(f"   Press Ctrl+C to stop\n")

    try:
        async for segment in pipeline.run():
            metrics = pipeline.get_metrics()
            print(
                f"\r⏱️  Latency: {metrics['e2e_p95_ms']:.0f}ms | "
                f"FPS: {metrics['fps']:.1f} | "
                f"Frames: {metrics['frames_processed']}",
                end=""
            )
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")

    final_metrics = pipeline.get_metrics()
    print(f"\n📊 Final Metrics:")
    print(f"   Frames processed: {final_metrics['frames_processed']}")
    print(f"   Average FPS: {final_metrics['fps']:.1f}")
    print(f"   E2E latency (p95): {final_metrics['e2e_p95_ms']:.0f}ms")
    print(f"   Errors: {final_metrics['errors']}")


async def cmd_dub(args):
    """Dub a video file."""
    from .pipeline import SimpleDubber

    input_path = args.input
    output_path = args.output or input_path.replace(".mp4", "_dubbed.mp4")

    print(f"🎬 Dubbing: {input_path}")
    print(f"   {args.source} → {args.target}")
    print(f"   Output: {output_path}\n")

    dubber = SimpleDubber(
        source_lang=args.source,
        target_lang=args.target,
        model_path=args.model
    )

    result = await dubber.dub_video(input_path, output_path)

    print(f"\n✅ Complete!")
    print(f"   Segments processed: {result['segments']}")
    print(f"   Output: {result['output']}")


def cmd_dub_file(args):
    """Run the file-based end-to-end dub (phase0 demo)."""
    from .filedub import dub_file, format_report, FFMPEG

    if FFMPEG is None:
        print("ERROR: ffmpeg not found. Set FFMPEG_BIN/FFPROBE_BIN or install ffmpeg.")
        sys.exit(1)

    output = args.output or args.input.rsplit(".", 1)[0] + "_dubbed.mp4"
    transcript = args.transcript
    if transcript is None and args.transcript_file:
        with open(args.transcript_file) as f:
            transcript = f.read().strip()

    print(f"Dubbing (file mode): {args.input} -> {output}")
    print(f"  {args.source} -> {args.target}")

    report = dub_file(
        input_video=args.input,
        output_video=output,
        source_lang=args.source,
        target_lang=args.target,
        transcript=transcript,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        tts_url=args.tts_url,
        tts_model=args.tts_model,
        tts_model_path=args.tts_model_path,
        use_musetalk=not args.no_musetalk,
        musetalk_repo=args.musetalk_repo,
        allow_tts_fallback=not args.no_tts_fallback,
    )
    print()
    print(format_report(report))


def cmd_status(args):
    """Show pipeline status."""
    print("🔍 Zen-Dub-Live Status")
    print("   No active sessions")


def cmd_voices(args):
    """List available anchor voices."""
    print("🎤 Available Anchor Voices:")
    print("   default - System default voice")
    print("   Upload custom voice with: zen-dub-live voice add <name> <audio.wav>")


if __name__ == "__main__":
    main()
