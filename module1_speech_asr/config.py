"""
config.py  —  Module 1 configuration constants
================================================
All tunable values for the Speech Capture & ASR pipeline live here.
No logic, no imports — just named constants.

Rule: if you find yourself typing a bare number anywhere in the pipeline
code, move it here first.
"""

import os
from pathlib import Path

# Load .env from the module1_speech_asr directory automatically.
# python-dotenv does nothing if the file doesn't exist, so this is safe.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not installed yet -- values must be set in the shell

# Suppress huggingface_hub symlink warning on Windows (cosmetic only)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# Audio ingestion
# ---------------------------------------------------------------------------

# Hard requirement — Whisper is trained exclusively on 16 kHz audio.
# Do NOT change this without resampling the input stream accordingly.
SAMPLE_RATE: int = 16_000          # Hz

# Number of audio channels expected from the capture device / stream.
CHANNELS: int = 1                  # mono

# Bit depth of the raw PCM frames.
SAMPLE_WIDTH: int = 2              # bytes (16-bit PCM)

# ---------------------------------------------------------------------------
# Streaming / polling cadence
# ---------------------------------------------------------------------------

# How often (seconds) the ASR engine re-transcribes the current buffer.
# Lower  → lower latency, higher CPU.  Start at 1.0 and tune from there.
POLL_INTERVAL_S: float = 1.0       # seconds

# LocalAgreement-N: how many consecutive ASR hypotheses must produce the
# same token prefix before that prefix is considered "committed" and
# handed off downstream (see Step 4 — asr_engine.py).
# N=2 is the value used in the Whisper-Streaming paper.
AGREEMENT_N: int = 2

# Safety cap: if uncommitted audio grows beyond this many seconds without
# a commit, flush whatever we have to prevent the buffer growing forever.
MAX_UNCOMMITTED_S: float = 15.0    # seconds

# ---------------------------------------------------------------------------
# Speaker diarization cadence
# ---------------------------------------------------------------------------

# Diarization is significantly slower than ASR polling, so it runs on a
# coarser cadence.  Increase if you see CPU pressure from pyannote.
DIARIZATION_INTERVAL_S: float = 4.0   # seconds

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

# faster-whisper model size.
# Options: "tiny", "base", "small", "medium", "large-v2", "large-v3"
# "small" is a good starting point for real-time use on a modern GPU.
WHISPER_MODEL_SIZE: str = "small"

# Compute backend for faster-whisper.
# "cpu" for CPU-only. Change to "cuda" if you have an NVIDIA GPU with CUDA installed.
WHISPER_DEVICE: str = "cpu"

# CTranslate2 quantisation.  "int8" is fastest; "float16" if you need
# extra accuracy on GPU; "float32" for CPU without int8 support.
WHISPER_COMPUTE_TYPE: str = "int8"

# Language hint fed to Whisper.  None → auto-detect (slower first chunk).
WHISPER_LANGUAGE: str | None = "en"

# Beam size used during decoding.  Larger = more accurate, slower.
WHISPER_BEAM_SIZE: int = 5

# pyannote.audio diarization pipeline model identifier on Hugging Face.
DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"

# ---------------------------------------------------------------------------
# Hugging Face access
# ---------------------------------------------------------------------------

# Name of the environment variable that holds the HF access token.
# The token must have access to the pyannote gated models.
# Set it in your shell:  export HF_TOKEN="hf_..."
HF_TOKEN_ENV_VAR: str = "HF_TOKEN"

# Convenience accessor — raises a clear error if the token is missing.
def get_hf_token() -> str:
    """Return the Hugging Face token from the environment.

    Raises
    ------
    EnvironmentError
        If the environment variable named by ``HF_TOKEN_ENV_VAR`` is not set.
    """
    token = os.environ.get(HF_TOKEN_ENV_VAR)
    if not token:
        raise EnvironmentError(
            f"Hugging Face token not found.  "
            f"Please set the {HF_TOKEN_ENV_VAR!r} environment variable.\n"
            f"Example:  export {HF_TOKEN_ENV_VAR}='hf_YOUR_TOKEN_HERE'"
        )
    return token

# ---------------------------------------------------------------------------
# WebSocket server (streaming output to Module 2)
# ---------------------------------------------------------------------------

# Host the WebSocket server binds to.
# Use "0.0.0.0" to accept connections from other machines on the network.
WS_HOST: str = "localhost"

# Port the WebSocket server listens on.
WS_PORT: int = 8765

# Maximum number of segments to buffer in the outbound queue before
# blocking the producer (back-pressure).
WS_QUEUE_MAX_SIZE: int = 100

# ---------------------------------------------------------------------------
# Sliding-window buffer
# ---------------------------------------------------------------------------

# When building the audio context fed to Whisper, include this many seconds
# of *already-committed* audio as left-context so word boundaries at chunk
# edges are decoded correctly.
CONTEXT_PADDING_S: float = 0.5    # seconds

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Python logging level for the module.  Change to "DEBUG" for verbose output.
LOG_LEVEL: str = "INFO"
