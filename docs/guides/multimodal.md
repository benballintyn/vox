# Multimodal (Vision + Video)

Send images and video alongside text. `ImageContent` and `VideoContent` are sibling content-part types — same `source_type` / `media_type` / `data` shape, different MIME prefix.

## Images

```python
from pathlib import Path
from vox import ImageContent, Message, TextContent, VoxClient

client = VoxClient()

message = Message(
    role="user",
    content=[
        TextContent(text="What's in this image?"),
        ImageContent(
            source_type="base64",
            media_type="image/png",
            data=Path("photo.png").read_bytes(),  # bytes auto-base64 encoded
        ),
    ],
)

response = client.complete([message], model="gpt-5")
print(response.message.text)
```

URL form:

```python
ImageContent(
    source_type="url",
    media_type="image/jpeg",
    data="https://example.com/photo.jpg",
)
```

Works on every provider that supports vision (OpenAI, Anthropic, Gemini, OpenRouter).

## Video

vox accepts video via `VideoContent` mirroring `ImageContent`'s shape:

| Provider | Behavior |
| -------- | -------- |
| Gemini   | **Native** — `inline_data` (base64) or `file_data` URI (incl. YouTube links). MIME: `video/mp4`, `video/webm`, `video/mov`, etc. |
| OpenAI / Anthropic / OpenRouter / LM Studio | **Fallback** — vox auto-extracts uniformly-sampled JPEG frames (default fps=1, max 8) via the `vox-llm[video]` extra, with a loud `loguru` warning per substitution so the cost implication stays visible. |

### Gemini native

```python
from pathlib import Path
from vox import Message, TextContent, VideoContent, VoxClient

client = VoxClient()

video = VideoContent(
    source_type="base64",
    media_type="video/mp4",
    data=Path("clip.mp4").read_bytes(),
)

response = client.complete(
    [
        Message(
            role="user",
            content=[
                TextContent(text="Summarize what happens in this clip."),
                video,
            ],
        )
    ],
    model="gemini-3.1-flash-lite",
)
```

Hosted-URI form (Gemini only — YouTube link or Files-API URI):

```python
VideoContent(
    source_type="url",
    media_type="video/mp4",
    data="https://www.youtube.com/watch?v=...",
)
```

### Frame-extraction fallback

Install the extra:

```bash
pip install "vox-llm[video]"
```

Then the same `VideoContent` works against OpenAI:

```python
response = client.complete(
    [Message(role="user", content=[TextContent(text="Describe"), video])],
    model="gpt-5",  # not native; vox extracts frames
)
# Emits a loguru WARNING:
#   "Provider openai has no native video input; auto-extracted 4
#    frames from video/mp4 video (fps=1, max=8). Image-token cost
#    will be ~4x a single image."
```

For explicit control, decode and pass `ImageContent` parts yourself instead of relying on the fallback.

## Audio

For audio input/output, vox uses dedicated `transcribe()` / `synthesize()` methods rather than bolting audio into `complete()`. See [Audio I/O](audio.md).

## See also

- [`ImageContent`](../reference/messages.md)
- [`VideoContent`](../reference/messages.md)
- [`Message.content`](../reference/messages.md) — accepts `str | list[ContentPart]`
