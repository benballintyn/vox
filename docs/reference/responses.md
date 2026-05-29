# Responses

Response types returned by `VoxClient` methods.

## CompletionResponse

Returned by `complete()` and `acomplete()`.

::: vox.CompletionResponse

## TranscriptionResponse

Returned by `transcribe()` and `atranscribe()`.

::: vox.TranscriptionResponse

## Usage

Token usage carried on `CompletionResponse.usage` and on streaming `usage` chunks.

::: vox.Usage

## StreamChunk

Yielded by `stream()` and `astream()`.

::: vox.StreamChunk

## Finish reasons

::: vox.FinishReason
    options:
      show_root_heading: false

::: vox.normalize_finish_reason
