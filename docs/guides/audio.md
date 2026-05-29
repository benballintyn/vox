# Audio I/O (transcribe + synthesize)

Audio doesn't fit naturally into the general `complete()` flow — the flagship reasoning models (Claude Opus / Sonnet, GPT-5, Gemini 3 Pro) don't accept audio natively. vox exposes audio through dedicated methods that hit each provider's actual STT / TTS surface.

| Provider | `transcribe()` (STT) | `synthesize()` (TTS) |
| -------- | -------------------- | -------------------- |
| OpenAI   | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` | `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` |
| Gemini   | `gemini-3.5-flash`+ via `generate_content` + audio Part | `gemini-3.1-flash-tts-preview` (PCM wrapped as WAV) |
| Anthropic / OpenRouter / LM Studio | raises `InvalidRequestError` | raises `InvalidRequestError` |

## Transcribe

```python
from pathlib import Path
from vox import AudioContent, VoxClient

client = VoxClient()

result = client.transcribe(
    AudioContent(
        source_type="base64",
        media_type="audio/wav",
        data=Path("meeting.wav").read_bytes(),  # bytes auto-base64 encoded
    ),
    model="whisper-1",
    language="en",              # ISO-639-1; OpenAI only, Gemini ignores
    prompt="meeting notes",     # optional bias prompt (Whisper)
)

print(result.text)
print(result.language, result.duration)   # populated when provider reports
```

## Synthesize

```python
import base64
from pathlib import Path

audio = client.synthesize(
    text="The quick brown fox jumps over the lazy dog.",
    voice="alloy",              # provider-specific values
    model="tts-1",              # or gpt-4o-mini-tts, gemini-3.1-flash-tts-preview
    format="mp3",               # OpenAI: mp3/opus/aac/flac/wav/pcm; Gemini: always wav
    speed=1.0,                  # OpenAI only
)

Path("out.mp3").write_bytes(base64.standard_b64decode(audio.data))
```

## Voices

- **OpenAI** (`vox.providers.openai.OPENAI_TTS_VOICES`): `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar`. `marin` and `cedar` are highest quality.
- **Gemini** (`vox.providers.gemini.GEMINI_TTS_VOICES`): `Aoede`, `Charon`, `Fenrir`, `Kore`, `Leda`, `Orus`, `Puck`, `Zephyr`.

## Async

```python
result = await client.atranscribe(audio, model="whisper-1")
audio = await client.asynthesize("Hello", voice="alloy", model="tts-1")
```

## Why dedicated methods

The flagship reasoning models — `gpt-5`, `claude-sonnet-4-5`, `gemini-3-pro` — don't accept audio natively. Only audio-tuned models do. Routing audio through `complete()` would have forced consumers to give up reasoning, structured output, and tool use whenever they touched audio. Separate methods that hit each provider's actual STT/TTS endpoints keep the abstraction honest.

## See also

- [`AudioContent`](../reference/messages.md)
- [`TranscriptionResponse`](../reference/responses.md)
- [`VoxClient.transcribe()`](../reference/client.md) / [`VoxClient.synthesize()`](../reference/client.md)
