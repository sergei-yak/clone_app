# Telegram Voice Clone Bot (Qwen3-TTS)

A Telegram bot that collects a user's voice sample + transcript, then generates cloned speech from future text prompts.

## Flow

1. User sends `/start`.
2. Bot asks for a voice sample.
3. Bot asks for transcript of that voice sample.
4. Bot initializes Qwen3-TTS and marks the session ready.
5. User sends any text and receives synthesized audio in cloned voice.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or explicitly: pip install python-telegram-bot==21.6
```

Set environment variables:

```bash
export TELEGRAM_BOT_TOKEN="<your bot token>"
export QWEN_TTS_MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
export QWEN_TTS_DEVICE="cuda:0"      # or cpu
export QWEN_TTS_DTYPE="bfloat16"     # float32/float16/bfloat16
export QWEN_TTS_ATTN="flash_attention_2"  # e.g. sdpa on CPU
```

## Run

```bash
python bot.py
```

## Windows troubleshooting

If you get `ImportError: cannot import name 'Update' from 'telegram'`, you installed the wrong package.

```powershell
pip uninstall -y telegram
pip install -r requirements.txt
# or explicitly: pip install python-telegram-bot==21.6
```

If you see `SoX could not be found!`, install SoX and add it to `PATH`.
If you see ffmpeg conversion errors, install ffmpeg and add it to `PATH`.

## Notes

- Telegram voice notes are converted to mono 24kHz WAV through `ffmpeg` before cloning.
- User session state is in-memory. Generated files are saved to `data/<telegram_user_id>/`.
- For production: persist session state and add queueing/rate limits.
