import asyncio
import importlib
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf
import torch
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("voice_clone_bot")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
MODEL_NAME = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
MODEL_DEVICE = os.getenv("QWEN_TTS_DEVICE", "cuda:0")
MODEL_DTYPE = os.getenv("QWEN_TTS_DTYPE", "bfloat16")
MODEL_ATTN = os.getenv("QWEN_TTS_ATTN", "flash_attention_2")
OUTPUT_DIR = Path(os.getenv("VOICE_CLONE_DATA_DIR", "data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_model_lock = asyncio.Lock()
_model: Optional[Any] = None


@dataclass
class UserSession:
    state: str = "await_voice"
    ref_audio_path: Optional[Path] = None
    ref_text: Optional[str] = None


sessions: Dict[int, UserSession] = {}


def _resolve_torch_dtype(dtype: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype.lower(), torch.bfloat16)


def _load_model_sync() -> Any:
    logger.info("Loading model %s on %s", MODEL_NAME, MODEL_DEVICE)
    qwen_tts = importlib.import_module("qwen_tts")
    model_cls = getattr(qwen_tts, "Qwen3TTSModel")
    return model_cls.from_pretrained(
        MODEL_NAME,
        device_map=MODEL_DEVICE,
        dtype=_resolve_torch_dtype(MODEL_DTYPE),
        attn_implementation=MODEL_ATTN,
    )


async def get_model() -> Any:
    global _model
    if _model is not None:
        return _model

    async with _model_lock:
        if _model is None:
            loop = asyncio.get_running_loop()
            _model = await loop.run_in_executor(None, _load_model_sync)
    return _model


def ensure_ffmpeg_installed() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert Telegram audio to wav.")


def convert_audio_to_wav(input_path: Path, output_path: Path) -> None:
    ensure_ffmpeg_installed()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "24000",
        "-ac",
        "1",
        str(output_path),
    ]
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {process.stderr.strip()}")


def generate_clone_sync(text: str, ref_audio: Path, ref_text: str, output_path: Path) -> Path:
    model = _model
    if model is None:
        raise RuntimeError("Model not loaded.")

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=str(ref_audio),
        ref_text=ref_text,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wavs[0], sr)
    return output_path


def dependency_help_message(exc: Exception) -> str:
    base = str(exc)
    return (
        "Model initialization failed. Ensure dependencies are installed correctly:\n"
        "1) pip uninstall -y telegram\n"
        "2) pip install -r requirements.txt\n"
        "3) install ffmpeg and sox, and ensure they are on PATH\n"
        f"\nOriginal error: {base}"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.message:
        return

    user_id = update.effective_user.id
    sessions[user_id] = UserSession()
    await update.message.reply_text(
        "Hi! Send me a voice sample (voice note/audio/document). Then send the exact transcript for that sample. "
        "After that I will clone your voice and generate speech from your text."
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.message:
        return

    user_id = update.effective_user.id
    session = sessions.setdefault(user_id, UserSession())

    file_obj = None
    ext = "bin"
    if update.message.voice:
        file_obj = await update.message.voice.get_file()
        ext = "ogg"
    elif update.message.audio:
        file_obj = await update.message.audio.get_file()
        file_name = update.message.audio.file_name or "audio.bin"
        ext = file_name.split(".")[-1]
    elif update.message.document:
        file_obj = await update.message.document.get_file()
        file_name = update.message.document.file_name or "doc.bin"
        ext = file_name.split(".")[-1]

    if file_obj is None:
        return

    user_dir = OUTPUT_DIR / str(user_id)
    raw_path = user_dir / f"reference_raw.{ext}"
    wav_path = user_dir / "reference.wav"

    await file_obj.download_to_drive(custom_path=str(raw_path))

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, convert_audio_to_wav, raw_path, wav_path)
    except Exception as exc:
        logger.exception("Failed to process audio")
        await update.message.reply_text(f"I could not process that audio: {exc}")
        return

    session.ref_audio_path = wav_path
    session.state = "await_transcript"
    await update.message.reply_text("Got your audio. Now send the transcript text for that audio sample.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    session = sessions.setdefault(user_id, UserSession())
    text = update.message.text.strip()

    if text.startswith("/"):
        return

    if session.state == "await_voice":
        await update.message.reply_text("Please send your voice sample first.")
        return

    if session.state == "await_transcript":
        session.ref_text = text
        session.state = "warming_up"
        await update.message.reply_text("Transcript saved. Initializing model, please wait...")
        try:
            await get_model()
        except Exception as exc:
            logger.exception("Model load failed")
            session.state = "await_transcript"
            await update.message.reply_text(dependency_help_message(exc))
            return

        session.state = "ready"
        await update.message.reply_text(
            "✅ Voice clone is ready. Send any text and I will generate cloned speech audio for you."
        )
        return

    if session.state != "ready" or not session.ref_audio_path or not session.ref_text:
        await update.message.reply_text("Run /start and complete setup first.")
        return

    await update.message.reply_chat_action(ChatAction.RECORD_VOICE)
    output_path = OUTPUT_DIR / str(user_id) / "output_voice_clone.wav"
    try:
        model = await get_model()
        if not model:
            raise RuntimeError("Model unavailable")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            generate_clone_sync,
            text,
            session.ref_audio_path,
            session.ref_text,
            output_path,
        )

        await update.message.reply_audio(audio=str(output_path), caption="Here is your cloned voice output.")
    except Exception as exc:
        logger.exception("Generation failed")
        await update.message.reply_text(f"Generation failed: {exc}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(
        "/start - begin setup\n"
        "1) send a voice sample\n"
        "2) send transcript\n"
        "3) send text to synthesize"
    )


def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN environment variable.")

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.ALL, handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Starting Telegram bot")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
