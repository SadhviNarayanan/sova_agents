"""
Voice utilities for CareRelay's in-house voice agent.

  text_to_speech  — converts a text string → MP3 bytes (ElevenLabs TTS)
  speech_to_text  — transcribes a complete audio file → text (ElevenLabs STT)
  realtime_agent  — live mic → ElevenLabs real-time STT WebSocket → TTS reply loop
"""

import asyncio
import io
import json
import os

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

load_dotenv()

# ── voice / model config (synced with ElevenLabs API docs) ──────────────────
VOICE_ID   = "JBFqnCBsd6RMkjVDRZzb"   # "George" — warm, authoritative
MODEL_TTS  = "eleven_multilingual_v2"
MODEL_STT  = "scribe_v1"
OUTPUT_FMT = "mp3_44100_128"

_ELEVENLABS_WSS = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


def _client() -> ElevenLabs:
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise EnvironmentError("ELEVENLABS_API_KEY is not set in .env")
    return ElevenLabs(api_key=key)


# ── TTS ──────────────────────────────────────────────────────────────────────

def text_to_speech(text: str, output_path: str | None = None) -> bytes:
    """Input: text string → Output: MP3 bytes spoken by George (empathetic tone)."""
    client = _client()

    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id=MODEL_TTS,
        output_format=OUTPUT_FMT,
    )

    audio_bytes = b"".join(audio_stream)

    if output_path:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

    return audio_bytes


# ── Batch STT ────────────────────────────────────────────────────────────────

def speech_to_text(audio: bytes | str, audio_format: str = "ogg") -> str:
    """Input: audio bytes or file path → Output: transcribed text string (ElevenLabs Scribe)."""
    client = _client()

    if isinstance(audio, str):
        with open(audio, "rb") as f:
            result = client.speech_to_text.convert(
                file=(f"audio.{audio_format}", f, f"audio/{audio_format}"),
                model_id=MODEL_STT,
            )
    else:
        result = client.speech_to_text.convert(
            file=(f"audio.{audio_format}", io.BytesIO(audio), f"audio/{audio_format}"),
            model_id=MODEL_STT,
        )

    return (result.text or "").strip()


# ── Real-time voice agent (WebSocket STT → API reply → TTS playback) ─────────

async def realtime_agent(get_reply, chunk_ms: int = 250) -> None:
    """
    Input: get_reply(text) callable that returns a response string (your API).
    Streams mic audio to ElevenLabs real-time STT WebSocket, calls get_reply on
    each completed utterance, speaks the response via TTS.

    Requires: pip install websockets sounddevice numpy
    """
    import numpy as np
    import sounddevice as sd
    import websockets

    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise EnvironmentError("ELEVENLABS_API_KEY is not set in .env")

    url = f"{_ELEVENLABS_WSS}?authorization={key}&model_id={MODEL_STT}"
    sample_rate = 16_000
    chunk_frames = int(sample_rate * chunk_ms / 1000)

    print("Voice agent ready — speak now. Ctrl-C to stop.")

    async with websockets.connect(url) as ws:
        loop = asyncio.get_event_loop()
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def _mic_callback(indata, _frames, _time, _status):
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            loop.call_soon_threadsafe(audio_queue.put_nowait, pcm)

        async def _send_audio():
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=chunk_frames,
                callback=_mic_callback,
            ):
                while True:
                    chunk = await audio_queue.get()
                    await ws.send(chunk)

        async def _recv_transcripts():
            async for raw in ws:
                event = json.loads(raw)
                if event.get("type") == "transcript" and event.get("is_final"):
                    user_text = event.get("text", "").strip()
                    if not user_text:
                        continue
                    print(f"You: {user_text}")
                    reply = (
                        await get_reply(user_text)
                        if asyncio.iscoroutinefunction(get_reply)
                        else get_reply(user_text)
                    )
                    print(f"Agent: {reply}")
                    audio = text_to_speech(reply)
                    play(audio)

        await asyncio.gather(_send_audio(), _recv_transcripts())


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sample = (
        " ".join(sys.argv[1:])
        or "The first move is what sets everything in motion."
    )

    print(f"Speaking: {sample!r}")
    audio = text_to_speech(sample)
    play(audio)
