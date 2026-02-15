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

## Notes

- Telegram voice notes are converted to mono 24kHz WAV through `ffmpeg` before cloning.
- User session state is in-memory. Generated files are saved to `data/<telegram_user_id>/`.
- For production: persist session state and add queueing/rate limits.

You can change voice model parameters to make it sound more/less natural, emotional, etc.:
```bash
GEN_TEMPERATURE = float(os.getenv("QWEN_TTS_TEMPERATURE", "0.8"))
GEN_TOP_P = float(os.getenv("QWEN_TTS_TOP_P", "0.99"))
GEN_TOP_K = int(os.getenv("QWEN_TTS_TOP_K", "150"))
GEN_REPETITION_PENALTY = float(os.getenv("QWEN_TTS_REPETITION_PENALTY", "1.08"))
```
