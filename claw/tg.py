"""
Telegram bot — text conversation interface for CareRelay.

Each chat has its own session. Users send text messages and the bot
replies via GPT-4o using the OpenAI key from the environment.

Required environment variables:
    TELEGRAM_BOT_TOKEN       — from @BotFather
    OPENAI_API_KEY           — OpenAI API key

Optional:
    TELEGRAM_ALLOWED_IDS env var — comma-separated chat IDs to whitelist
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Union

from openai import OpenAI
from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

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

SYSTEM_PROMPT = (
    "You are CareRelay, a compassionate AI care assistant helping patients "
    "recover after a hospital stay. Keep responses concise, warm, and clinically "
    "aware. If a patient describes urgent symptoms, advise them to call 911 or "
    "their care team immediately."
)


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY env var is required.")
    return OpenAI(api_key=api_key)


def _call_backend(message: str, history: list) -> str:
    client = _get_openai_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


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
# Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return
    _session(chat_id)
    await update.message.reply_text(
        "Hi, I'm CareRelay. I'm here to support you after your hospital stay. "
        "How are you feeling today?"
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
        reply = _call_backend(user_text, session["history"])
    except Exception as exc:
        log.error("Backend error: %s", exc)
        await update.message.reply_text(
            "I'm having trouble reaching the care system right now. Please try again in a moment."
        )
        return

    session["history"].append({"role": "assistant", "content": reply})
    await update.message.reply_text(reply)


# ---------------------------------------------------------------------------
# send_message helpers
# ---------------------------------------------------------------------------

def send_message(bot_token: str, chat_id: Union[str, int], text: str) -> dict:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": str(chat_id), "text": text, "parse_mode": "HTML"}
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def send_message_from_env(message: str) -> dict:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars are required.")
    return send_message(bot_token, chat_id, message)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN env var is required.")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("CareRelay Telegram bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
