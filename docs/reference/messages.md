# Messages & Content

The shape of conversation messages and the typed content parts that go inside them.

## Message

::: vox.Message

## Content parts

`Message.content` accepts either a plain string or a `list[ContentPart]` for multimodal input. `ContentPart` is the union of the content types below.

::: vox.TextContent

::: vox.ImageContent

::: vox.VideoContent

::: vox.AudioContent

## Tool calls

::: vox.ToolCallData
