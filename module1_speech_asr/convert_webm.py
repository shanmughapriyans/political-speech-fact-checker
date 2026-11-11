"""
convert_webm.py — Convert .webm (or any format av can read) to 16-kHz mono WAV
using PyAV (already installed as a faster-whisper dependency).

Usage:
    python convert_webm.py input.webm output.wav
"""
import sys
import struct
import wave
import array

import av

def convert(src: str, dst: str, sample_rate: int = 16000) -> None:
    container = av.open(src)
    stream = next(s for s in container.streams if s.type == "audio")

    resampler = av.AudioResampler(
        format="s16",          # signed 16-bit PCM
        layout="mono",
        rate=sample_rate,
    )

    pcm_data: list[bytes] = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            resampled = resampler.resample(frame)
            for rf in resampled:
                pcm_data.append(bytes(rf.planes[0]))

    # Flush resampler
    for rf in resampler.resample(None):
        pcm_data.append(bytes(rf.planes[0]))

    container.close()

    raw = b"".join(pcm_data)
    with wave.open(dst, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)        # 2 bytes = 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(raw)

    duration = len(raw) / (sample_rate * 2)
    print(f"Converted: {src} -> {dst}  ({duration:.1f}s, {sample_rate}Hz mono)")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "demo_speech.webm"
    dst = sys.argv[2] if len(sys.argv) > 2 else "demo_speech.wav"
    convert(src, dst)

