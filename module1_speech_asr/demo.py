"""
demo.py  --  Module 1 live console demo
========================================
Runs the full pipeline (real faster-whisper + optional diarization)
and prints a colour-coded, live-updating transcript to the terminal.

Usage
-----
    python demo.py --download-sample          # grabs a sample, transcribes it
    python demo.py --file speech.wav          # use your own WAV
    python demo.py --file speech.wav --diarize
    python demo.py --mic                      # live microphone
    python demo.py --download-sample --model tiny  # fastest for testing

Colour legend
-------------
    YELLOW  = tentative (ASR still thinking)
    GREEN   = final commit (ASR confident)
    CYAN    = speaker label (diarization on)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

# Force UTF-8 output on Windows so we never hit cp1252 crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# ANSI colour helpers (no third-party deps)
# ---------------------------------------------------------------------------
RESET      = "\033[0m"
BOLD       = "\033[1m"
DIM        = "\033[2m"
YELLOW     = "\033[93m"
GREEN      = "\033[92m"
CYAN       = "\033[96m"
BLUE_BG    = "\033[44m"
CLEAR_LINE = "\033[2K\r"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _banner() -> None:
    print()
    print(_c("  Module 1 - Speech Capture & ASR  ", BOLD, BLUE_BG))
    print(_c("  LocalAgreement-n | faster-whisper | pyannote  ", DIM))
    print()
    print(f"  {_c('[~]', YELLOW)} tentative   "
          f"{_c('[+]', GREEN)} final   "
          f"{_c('[S]', CYAN)} speaker label")
    print("  " + "-" * 52)
    print()


# ---------------------------------------------------------------------------
# Sample audio download
# ---------------------------------------------------------------------------

SAMPLE_URL  = "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav"
SAMPLE_PATH = Path(__file__).parent / "sample_gettysburg.wav"


def download_sample() -> Path:
    if SAMPLE_PATH.exists():
        print(f"[sample] Using cached: {SAMPLE_PATH.name}")
        return SAMPLE_PATH
    print(f"[sample] Downloading public-domain speech sample (~10 s) ...")
    print(f"         {SAMPLE_URL}")
    try:
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
        print(f"[sample] Saved: {SAMPLE_PATH.name}\n")
    except Exception as e:
        print(f"[sample] Download failed: {e}")
        print("[sample] Supply your own WAV with:  python demo.py --file audio.wav")
        sys.exit(1)
    return SAMPLE_PATH


# ---------------------------------------------------------------------------
# Resample to 16 kHz if needed
# ---------------------------------------------------------------------------

def _ensure_16k(path: Path) -> Path:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    if sr == 16000:
        return path
    print(f"[resample] File is {sr} Hz -- resampling to 16 kHz ...")
    data_mono = data.mean(axis=1) if data.ndim == 2 else data
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(16000, sr)
        data_16k = resample_poly(data_mono, 16000 // g, sr // g).astype(np.float32)
    except ImportError:
        n_out    = int(len(data_mono) * 16000 / sr)
        data_16k = np.interp(
            np.linspace(0, len(data_mono), n_out),
            np.arange(len(data_mono)),
            data_mono,
        ).astype(np.float32)
    out = path.with_stem(path.stem + "_16k")
    sf.write(str(out), data_16k, 16000)
    print(f"[resample] Saved: {out.name}\n")
    return out


# ---------------------------------------------------------------------------
# on_event callback -- colour printer
# ---------------------------------------------------------------------------

_last_tent = False


def make_on_event(show_tentative: bool = True):
    def on_event(seg) -> None:
        global _last_tent
        ts = f"[{seg.start:5.1f}s-{seg.end:5.1f}s]"

        if seg.is_final:
            if _last_tent:
                print(CLEAR_LINE, end="")
            spk = (f" {_c(seg.speaker, CYAN)}" if seg.speaker != "UNKNOWN" else "")
            print(f"  {_c(ts, DIM)}{spk}  {_c(seg.text, GREEN, BOLD)}")
            _last_tent = False
        else:
            if show_tentative:
                spk = (f" {_c(seg.speaker, CYAN)}" if seg.speaker != "UNKNOWN" else "")
                print(f"{CLEAR_LINE}  {_c(ts, DIM)}{spk}  {_c(seg.text + ' ...', YELLOW)}",
                      end="", flush=True)
                _last_tent = True

    return on_event


# ---------------------------------------------------------------------------
# Spinner helper
# ---------------------------------------------------------------------------

def _make_spinner(label: str):
    """Return (start_fn, stop_fn) for a simple ASCII spinner."""
    stop_ev = threading.Event()
    chars   = ["-", "\\", "|", "/"]

    def _run():
        i = 0
        while not stop_ev.is_set():
            print(f"\r  {label} {chars[i % 4]}", end="", flush=True)
            time.sleep(0.12)
            i += 1

    t = threading.Thread(target=_run, daemon=True)

    def start():
        t.start()

    def stop(done_msg: str):
        stop_ev.set()
        t.join()
        print(f"\r  {_c(done_msg, GREEN)}" + " " * 20)

    return start, stop


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Module 1 live console demo")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--file",            metavar="PATH", help="Path to a WAV file")
    src.add_argument("--mic",             action="store_true", help="Live microphone")
    src.add_argument("--download-sample", action="store_true",
                     help="Download a public-domain sample WAV and transcribe it")
    p.add_argument("--diarize",      action="store_true", help="Enable speaker diarization")
    p.add_argument("--model",        default=None,
                   help="Whisper model: tiny|base|small|medium|large-v2 (default: from config)")
    p.add_argument("--language",     default=None,
                   metavar="LANG",
                   help="Language code: 'en', 'hi', 'fr', 'auto', etc. "
                        "Default: 'en'. Use 'auto' for auto-detection (slower first chunk). "
                        "See: https://github.com/openai/whisper#available-models-and-languages")
    p.add_argument("--no-tentative", action="store_true",
                   help="Hide yellow tentative output, show final only")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Default behaviour: download sample if nothing specified
    if not args.file and not args.mic and not args.download_sample:
        args.download_sample = True

    _banner()

    # Resolve audio source
    if args.download_sample:
        wav_path = download_sample()
        wav_path = _ensure_16k(wav_path)
        args.file = str(wav_path)

    # Deferred imports so --help is instant
    import config
    from audio_capture import FileAudioCapture, MicAudioCapture
    from diarization import Diarizer
    from pipeline import Pipeline
    from streaming_asr import StreamingASR

    if args.model:
        config.WHISPER_MODEL_SIZE = args.model

    # Language override
    if args.language:
        if args.language.lower() == "auto":
            config.WHISPER_LANGUAGE = None   # None = auto-detect
            print(f"  Language: {_c('auto-detect', BOLD)}")
        else:
            config.WHISPER_LANGUAGE = args.language
            print(f"  Language: {_c(args.language, BOLD)}")
    else:
        print(f"  Language: {_c(config.WHISPER_LANGUAGE or 'auto-detect', BOLD)}")

    # Print config summary
    print(f"  Model   : {_c(config.WHISPER_MODEL_SIZE, BOLD)}")
    if args.file:
        print(f"  Source  : {_c(args.file, BOLD)}")
        capture = FileAudioCapture(args.file, chunk_s=0.5, realtime=True)
    else:
        print(f"  Source  : {_c('microphone', BOLD)}")
        capture = MicAudioCapture(chunk_s=0.5)
    print(f"  Diarize : {_c(str(args.diarize), BOLD)}")
    print()

    # Load faster-whisper
    spin_start, spin_stop = _make_spinner("Loading faster-whisper model")
    spin_start()
    asr = StreamingASR()
    spin_stop("faster-whisper ready")

    # Load pyannote (optional)
    diarizer = None
    if args.diarize:
        spin_start2, spin_stop2 = _make_spinner("Loading pyannote diarization pipeline")
        spin_start2()
        diarizer = Diarizer()
        spin_stop2("pyannote diarization ready")

    print()
    print("  " + "-" * 52)
    print(f"  {_c('Live transcript:', BOLD)}")
    print()

    on_event = make_on_event(show_tentative=not args.no_tentative)

    pipeline = Pipeline(
        capture=capture,
        asr=asr,
        diarizer=diarizer,
        on_event=on_event,
    )

    t0 = time.monotonic()
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\n  [stopped by user]")

    elapsed = time.monotonic() - t0
    print()
    print("  " + "-" * 52)
    print(f"  {_c('Done', GREEN)}  ({elapsed:.1f} s total)")
    print()


if __name__ == "__main__":
    main()
