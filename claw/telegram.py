"""
Telegram bot — voice conversation interface for CareRelay.

Each chat has its own session. Users can send text or voice messages;
the bot transcribes audio (Whisper), calls the backend API for a reply,
and responds with both text and a voice message (ElevenLabs).

Required env vars:
    TELEGRAM_BOT_TOKEN   — from @BotFather
    CHAT_API_URL         — backend endpoint, e.g. http://localhost:8000/chat
                           POST {"message": str, "session_id": str}
                           expects {"reply": str}
    ELEVENLABS_API_KEY
    OPENAI_API_KEY

Optional:
    TELEGRAM_ALLOWED_IDS — comma-separated chat IDs to whitelist (leave unset to allow all)
"""

import io
import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Union

from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from voice import speech_to_text, text_to_speech

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_sessions: dict[int, dict] = {}


def _session(chat_id: int) -> dict:
    if chat_id not in _sessions:
        _sessions[chat_id] = {"chat_id": chat_id, "history": []}
    return _sessions[chat_id]


# ---------------------------------------------------------------------------
# Backend API
# ---------------------------------------------------------------------------

def _call_backend(message: str, session_id: str) -> str:
    """POST to CHAT_API_URL and return the reply text."""
    url = os.environ.get("CHAT_API_URL", "").rstrip("/")
    if not url:
        raise EnvironmentError("CHAT_API_URL is not set.")

    payload = json.dumps({"message": message, "session_id": session_id}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    reply = data.get("reply") or data.get("response") or data.get("text")
    if not reply:
        raise ValueError(f"Backend returned no reply field: {data}")
    return reply


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------

def _allowed(chat_id: int) -> bool:
    raw = os.environ.get("TELEGRAM_ALLOWED_IDS", "").strip()
    if not raw:
        return True
    allowed = {int(x.strip()) for x in raw.split(",") if x.strip()}
    return chat_id in allowed


# ---------------------------------------------------------------------------
# Shared reply helper
# ---------------------------------------------------------------------------

async def _reply(update: Update, text: str) -> None:
    """Send a text message and a voice message with the same content."""
    await update.message.reply_text(text)
    try:
        audio = text_to_speech(text)
        await update.message.reply_voice(
            voice=io.BytesIO(audio),
            caption="🔊",
        )
    except Exception as exc:
        log.warning("TTS failed, skipping voice reply: %s", exc)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return

    _session(chat_id)
    await _reply(
        update,
        "Hi, I'm CareRelay. I'm here to support you after your hospital stay. "
        "You can type or send a voice message — I'll listen and respond. "
        "How are you feeling today?",
    )


async def cmd_reset(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return
    _sessions.pop(chat_id, None)
    await update.message.reply_text("Session reset. Say hello whenever you're ready.")


async def handle_text(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return

    user_text = update.message.text.strip()
    if not user_text:
        return

    session = _session(chat_id)
    session["history"].append({"role": "user", "content": user_text})

    await update.message.chat.send_action(constants.ChatAction.TYPING)

    try:
        reply = _call_backend(user_text, str(chat_id))
    except Exception as exc:
        log.error("Backend error: %s", exc)
        await update.message.reply_text(
            "I'm having trouble reaching the care system right now. Please try again in a moment."
        )
        return

    session["history"].append({"role": "assistant", "content": reply})
    await _reply(update, reply)


async def handle_voice(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return

    await update.message.chat.send_action(constants.ChatAction.RECORD_VOICE)

    # Download the OGG voice file Telegram sends
    voice_file = await update.message.voice.get_file()
    ogg_bytes = await voice_file.download_as_bytearray()

    try:
        user_text = speech_to_text(bytes(ogg_bytes), audio_format="ogg")
    except Exception as exc:
        log.error("STT error: %s", exc)
        await update.message.reply_text("I couldn't make out that audio. Could you try again?")
        return

    if not user_text:
        await update.message.reply_text("I didn't catch anything in that message — please try again.")
        return

    # Echo back what was heard so the user can confirm
    await update.message.reply_text(f'I heard: "{user_text}"')

    session = _session(chat_id)
    session["history"].append({"role": "user", "content": user_text})

    await update.message.chat.send_action(constants.ChatAction.TYPING)

    try:
        reply = _call_backend(user_text, str(chat_id))
    except Exception as exc:
        log.error("Backend error: %s", exc)
        await update.message.reply_text(
            "I'm having trouble reaching the care system right now. Please try again in a moment."
        )
        return

    session["history"].append({"role": "assistant", "content": reply})
    await _reply(update, reply)


# ---------------------------------------------------------------------------
# Legacy send_message helpers (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def send_message(bot_token: str, chat_id: Union[str, int], text: str) -> dict:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": str(chat_id), "text": text, "parse_mode": "HTML"}
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def send_message_from_config(message: str, config_path: Union[str, Path] = None) -> dict:
    path = Path(config_path) if config_path else Path(__file__).parent / "telegram_config.json"
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f).get("telegram", {})
    return send_message(cfg["bot_token"], cfg["chat_id"], message)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN is not set.")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    log.info("CareRelay Telegram bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
