"""Tests for the OpenAI Responses API provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from vox import Message, OpenAIReasoning, ProviderConfig, ReasoningConfig, Tool
from vox.models.messages import ToolCallData
from vox.providers.openai import OpenAIProvider

from .conftest import make_openai_responses_api_response


@pytest.fixture
def provider() -> OpenAIProvider:
    """Create an OpenAI provider with a test API key."""
    return OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-4o"))


class TestMessageTranslation:
    """Tests for Responses API message translation."""

    def test_simple_user_message(self, provider: OpenAIProvider) -> None:
        messages = [Message(role="user", content="Hello")]
        items, _ = provider._translate_input(messages)
        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert items[0]["type"] == "message"
        assert items[0]["content"][0]["type"] == "input_text"
        assert items[0]["content"][0]["text"] == "Hello"

    def test_system_message_becomes_instructions(self, provider: OpenAIProvider) -> None:
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
        ]
        items, instructions = provider._translate_input(messages)
        assert instructions == "You are helpful."
        assert len(items) == 1  # System not in items

    def test_tool_result_message(self, provider: OpenAIProvider) -> None:
        messages = [
            Message(role="tool", content="72F", tool_call_id="call_1", name="weather"),
        ]
        items, _ = provider._translate_input(messages)
        assert items[0]["type"] == "function_call_output"
        assert items[0]["call_id"] == "call_1"
        assert items[0]["output"] == "72F"

    def test_assistant_with_tool_calls(self, provider: OpenAIProvider) -> None:
        messages = [
            Message(
                role="assistant",
                content="Checking weather.",
                tool_calls=[
                    ToolCallData(id="call_1", name="weather", arguments={"city": "NYC"}),
                ],
            ),
        ]
        items, _ = provider._translate_input(messages)
        # Text + function_call
        assert any(i.get("type") == "message" for i in items)
        assert any(i.get("type") == "function_call" for i in items)

    def test_assistant_tool_call_uses_fc_id_from_provider_state(
        self, provider: OpenAIProvider
    ) -> None:
        """vox#17 — Responses API requires fc_* on ``input[*].id``.

        When a previously-issued ToolCallData is sent back, the
        ``id`` field on the outbound function_call item must be the
        original ``fc_*`` (function_call item ID), not the ``call_*``
        cross-turn reference. vox preserves the former in
        ``provider_state["openai_fc_id"]``.
        """
        messages = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCallData(
                        id="call_xyz",
                        name="weather",
                        arguments={"city": "NYC"},
                        provider_state={"openai_fc_id": "fc_xyz"},
                    ),
                ],
            ),
        ]
        items, _ = provider._translate_input(messages)
        fc_item = next(i for i in items if i.get("type") == "function_call")
        assert fc_item["id"] == "fc_xyz", "outbound id must be the fc_* value"
        assert fc_item["call_id"] == "call_xyz", "call_id must remain the public id"

    def test_assistant_tool_call_replays_reasoning_item(self, provider: OpenAIProvider) -> None:
        """vox#25 — outbound emits the buffered reasoning item before fc.

        Reasoning models (gpt-5) emit a ``reasoning`` item before each
        function_call, and the Responses API rejects the assistant
        message on subsequent turns if that reasoning item isn't
        replayed. vox stashes it in
        ``ToolCallData.provider_state["openai_reasoning_item"]``; the
        outbound translator emits it as a peer item just before the
        ``function_call`` item.
        """
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_abc",
            "encrypted_content": "opaque-encrypted-context",
            "summary": [{"type": "summary_text", "text": "Looking up weather."}],
        }
        messages = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCallData(
                        id="call_abc",
                        name="weather",
                        arguments={"city": "NYC"},
                        provider_state={
                            "openai_fc_id": "fc_abc",
                            "openai_reasoning_item": reasoning_item,
                        },
                    ),
                ],
            ),
        ]
        items, _ = provider._translate_input(messages)
        types = [i.get("type") for i in items]
        assert "reasoning" in types
        assert "function_call" in types
        reasoning_idx = types.index("reasoning")
        fc_idx = types.index("function_call")
        assert reasoning_idx < fc_idx, "reasoning item must precede the function_call"
        assert items[reasoning_idx] == reasoning_item, "reasoning item replayed verbatim"
        assert items[fc_idx]["id"] == "fc_abc"
        assert items[fc_idx]["call_id"] == "call_abc"

    def test_assistant_tool_call_no_reasoning_item_omits_it(
        self, provider: OpenAIProvider
    ) -> None:
        """When provider_state has no reasoning item, only function_call is emitted."""
        messages = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCallData(
                        id="call_abc",
                        name="weather",
                        arguments={"city": "NYC"},
                        provider_state={"openai_fc_id": "fc_abc"},
                    ),
                ],
            ),
        ]
        items, _ = provider._translate_input(messages)
        assert not any(i.get("type") == "reasoning" for i in items)
        assert any(i.get("type") == "function_call" for i in items)

    def test_assistant_tool_call_falls_back_to_id_without_provider_state(
        self, provider: OpenAIProvider
    ) -> None:
        """Tool calls built from scratch (no provider_state) fall back to ``tc.id``.

        These flows aren't round-tripping a previously-issued call, so
        there's no original fc_id to preserve — using ``tc.id`` for both
        fields matches the pre-vox#17 behavior.
        """
        messages = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCallData(id="call_xyz", name="weather", arguments={}),
                ],
            ),
        ]
        items, _ = provider._translate_input(messages)
        fc_item = next(i for i in items if i.get("type") == "function_call")
        assert fc_item["id"] == "call_xyz"
        assert fc_item["call_id"] == "call_xyz"


class TestToolTranslation:
    """Tests for Responses API tool translation."""

    def test_tool_format(self, provider: OpenAIProvider) -> None:
        tools = [
            Tool(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            ),
        ]
        result = provider._translate_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "search"
        assert result[0]["parameters"]["type"] == "object"


class TestResponseTranslation:
    """Tests for Responses API response translation."""

    def test_text_response(self, provider: OpenAIProvider) -> None:
        mock_resp = make_openai_responses_api_response(content="Hello world!")
        result = provider._translate_response(mock_resp, "gpt-4o")
        assert result.message.text == "Hello world!"
        assert result.provider == "openai"
        assert result.model == "gpt-4o"
        assert result.usage.prompt_tokens == 10

    def test_function_call_response(self, provider: OpenAIProvider) -> None:
        mock_resp = make_openai_responses_api_response(
            content="",
            function_calls=[
                {"id": "call_1", "name": "weather", "arguments": {"city": "NYC"}},
            ],
        )
        result = provider._translate_response(mock_resp, "gpt-4o")
        assert result.message.tool_calls is not None
        assert len(result.message.tool_calls) == 1
        assert result.message.tool_calls[0].name == "weather"
        assert result.message.tool_calls[0].arguments == {"city": "NYC"}

    def test_reasoning_item_attached_to_following_function_call(
        self, provider: OpenAIProvider
    ) -> None:
        """vox#25 — the most recent reasoning item rides on the next fc.

        When the response output contains a ``reasoning`` item followed
        by a ``function_call`` item, vox stashes a faithful dict copy
        of the reasoning item in the ToolCallData's provider_state so
        the outbound translator can replay it. The buffer is consumed
        after a single attachment, so a *second* function_call without
        an intervening reasoning item does NOT receive the same item.
        """
        from unittest.mock import MagicMock

        # Simulate a Pydantic-shaped SDK item: model_dump returns the
        # full dict the outbound side should replay verbatim.
        reasoning_payload = {
            "type": "reasoning",
            "id": "rs_xyz",
            "encrypted_content": "ENCRYPTED",
            "summary": [{"type": "summary_text", "text": "Thinking..."}],
        }
        reasoning_item = MagicMock()
        reasoning_item.type = "reasoning"
        reasoning_item.summary = []  # no thinking_block leak in this test
        reasoning_item.model_dump.return_value = reasoning_payload

        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.id = "fc_xyz"
        fc_item.call_id = "call_xyz"
        fc_item.name = "weather"
        fc_item.arguments = '{"city": "NYC"}'

        # Second function_call WITHOUT an intervening reasoning item —
        # the buffer is consumed by the first, so this one's
        # provider_state should not carry a reasoning item.
        fc_item_2 = MagicMock()
        fc_item_2.type = "function_call"
        fc_item_2.id = "fc_xyz2"
        fc_item_2.call_id = "call_xyz2"
        fc_item_2.name = "weather"
        fc_item_2.arguments = '{"city": "LA"}'

        usage = MagicMock()
        usage.input_tokens = 5
        usage.output_tokens = 3
        usage.reasoning_tokens = 4

        mock_resp = MagicMock()
        mock_resp.output = [reasoning_item, fc_item, fc_item_2]
        mock_resp.usage = usage
        mock_resp.status = "completed"
        mock_resp.id = "resp_test"
        mock_resp.incomplete_details = None

        result = provider._translate_response(mock_resp, "gpt-5-mini")
        assert result.message.tool_calls is not None
        assert len(result.message.tool_calls) == 2
        tc1, tc2 = result.message.tool_calls

        assert tc1.provider_state is not None
        assert tc1.provider_state["openai_fc_id"] == "fc_xyz"
        assert tc1.provider_state["openai_reasoning_item"] == reasoning_payload

        assert tc2.provider_state is not None
        assert tc2.provider_state["openai_fc_id"] == "fc_xyz2"
        assert "openai_reasoning_item" not in tc2.provider_state

    def test_function_call_without_preceding_reasoning_has_no_item(
        self, provider: OpenAIProvider
    ) -> None:
        """No reasoning item ⇒ provider_state carries only openai_fc_id."""
        from unittest.mock import MagicMock

        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.id = "fc_abc"
        fc_item.call_id = "call_abc"
        fc_item.name = "weather"
        fc_item.arguments = "{}"

        usage = MagicMock()
        usage.input_tokens = 5
        usage.output_tokens = 3
        usage.reasoning_tokens = 0

        mock_resp = MagicMock()
        mock_resp.output = [fc_item]
        mock_resp.usage = usage
        mock_resp.status = "completed"
        mock_resp.id = "resp_test"
        mock_resp.incomplete_details = None

        result = provider._translate_response(mock_resp, "gpt-4o")
        assert result.message.tool_calls is not None
        tc = result.message.tool_calls[0]
        assert tc.provider_state is not None
        assert "openai_reasoning_item" not in tc.provider_state

    def test_function_call_captures_both_ids(self, provider: OpenAIProvider) -> None:
        """vox#17 — capture the distinct fc_* and call_* IDs from the response.

        The Responses API emits each function call with two IDs:
        ``id`` (``fc_*``, the output-item ID) and ``call_id`` (``call_*``,
        the cross-turn reference). vox exposes ``call_id`` as the public
        ``ToolCallData.id`` (consumers reference it in tool result
        messages) and stashes ``fc_id`` in ``provider_state`` for
        round-tripping. The default mock helper sets both to the same
        string, so this test builds the mock directly.
        """
        from unittest.mock import MagicMock

        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.id = "fc_abc"
        fc_item.call_id = "call_abc"
        fc_item.name = "weather"
        fc_item.arguments = '{"city": "NYC"}'

        usage = MagicMock()
        usage.input_tokens = 5
        usage.output_tokens = 3
        usage.reasoning_tokens = 0

        mock_resp = MagicMock()
        mock_resp.output = [fc_item]
        mock_resp.usage = usage
        mock_resp.status = "completed"
        mock_resp.id = "resp_test"
        mock_resp.incomplete_details = None

        result = provider._translate_response(mock_resp, "gpt-4o")
        assert result.message.tool_calls is not None
        tc = result.message.tool_calls[0]
        assert tc.id == "call_abc", "public id is the call_* value"
        assert tc.provider_state == {"openai_fc_id": "fc_abc"}


class TestComplete:
    """Tests for the complete method with mocked SDK."""

    def test_sync_complete(self, provider: OpenAIProvider, mocker) -> None:
        mock_resp = make_openai_responses_api_response(content="Hi there!")
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp
        provider._sync_client = mock_client

        result = provider.complete(
            [Message(role="user", content="Hello")],
            model="gpt-4o",
        )
        assert result.message.text == "Hi there!"
        mock_client.responses.create.assert_called_once()

    async def test_async_complete(self, provider: OpenAIProvider, mocker) -> None:
        mock_resp = make_openai_responses_api_response(content="Hi there!")
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)
        provider._async_client = mock_client

        result = await provider.acomplete(
            [Message(role="user", content="Hello")],
            model="gpt-4o",
        )
        assert result.message.text == "Hi there!"


class TestReasoningTranslation:
    """Tests for ReasoningConfig → Responses API reasoning param."""

    def test_semantic_level_maps_to_effort(self, provider: OpenAIProvider) -> None:
        request = provider._build_request_kwargs(
            [Message(role="user", content="Hi")],
            model="gpt-5",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=ReasoningConfig(level="medium"),
            stop=None,
        )
        assert request["reasoning"] == {"effort": "medium"}

    def test_minimal_level_supported(self, provider: OpenAIProvider) -> None:
        """GPT-5's 'minimal' effort tier is reachable through the semantic level."""
        request = provider._build_request_kwargs(
            [Message(role="user", content="Hi")],
            model="gpt-5",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=ReasoningConfig(level="minimal"),
            stop=None,
        )
        assert request["reasoning"] == {"effort": "minimal"}

    def test_openai_override_with_xhigh(self, provider: OpenAIProvider) -> None:
        """xhigh is OpenAI-specific and only reachable via the override."""
        request = provider._build_request_kwargs(
            [Message(role="user", content="Hi")],
            model="gpt-5",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=ReasoningConfig(openai=OpenAIReasoning(effort="xhigh")),
            stop=None,
        )
        assert request["reasoning"]["effort"] == "xhigh"

    def test_openai_override_with_summary(self, provider: OpenAIProvider) -> None:
        """summary param flows through for thinking blocks to be returned."""
        request = provider._build_request_kwargs(
            [Message(role="user", content="Hi")],
            model="gpt-5",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=ReasoningConfig(
                openai=OpenAIReasoning(effort="high", summary="auto"),
            ),
            stop=None,
        )
        assert request["reasoning"]["effort"] == "high"
        assert request["reasoning"]["summary"] == "auto"

    def test_override_takes_priority_over_level(self, provider: OpenAIProvider) -> None:
        request = provider._build_request_kwargs(
            [Message(role="user", content="Hi")],
            model="gpt-5",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=ReasoningConfig(
                level="low",
                openai=OpenAIReasoning(effort="xhigh"),
            ),
            stop=None,
        )
        assert request["reasoning"]["effort"] == "xhigh"

    def test_disabled_reasoning_omits_param(self, provider: OpenAIProvider) -> None:
        request = provider._build_request_kwargs(
            [Message(role="user", content="Hi")],
            model="gpt-5",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=ReasoningConfig(enabled=False, level="high"),
            stop=None,
        )
        assert "reasoning" not in request

    def test_no_reasoning_omits_param(self, provider: OpenAIProvider) -> None:
        request = provider._build_request_kwargs(
            [Message(role="user", content="Hi")],
            model="gpt-5",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=None,
            stop=None,
        )
        assert "reasoning" not in request


class TestVideoFallback:
    """VideoContent on OpenAI is auto-converted to image frames + warning."""

    def test_video_replaced_with_image_frames(self, provider: OpenAIProvider) -> None:
        from vox import ImageContent, Message, TextContent, VideoContent

        # Send video; substitute helper turns it into image frames; translation
        # emits input_image parts to the Responses API.
        messages = [
            Message(
                role="user",
                content=[
                    TextContent(text="describe"),
                    VideoContent(data="ZmFrZQ==", media_type="video/mp4"),
                ],
            )
        ]

        # Patch the substitute helper at its definition site so the
        # local-import in the provider resolves to the patched object.
        from unittest.mock import patch as _patch

        fake_frames = [
            ImageContent(data="ZnJhbWUx", media_type="image/jpeg"),
            ImageContent(data="ZnJhbWUy", media_type="image/jpeg"),
        ]
        with _patch(
            "vox._video.substitute_video_with_frames",
            return_value=[
                TextContent(text="describe"),
                *fake_frames,
            ],
        ) as sub:
            items, _ = provider._translate_input(messages)

        sub.assert_called_once()
        # provider_name should be forwarded.
        assert sub.call_args.kwargs["provider_name"] == "openai"

        # Resulting content has 1 text + 2 image parts.
        content = items[0]["content"]
        text_parts = [p for p in content if p["type"] == "input_text"]
        image_parts = [p for p in content if p["type"] == "input_image"]
        assert len(text_parts) == 1
        assert len(image_parts) == 2
        assert image_parts[0]["image_url"].startswith("data:image/jpeg;base64,")


class TestOpenAITranscribe:
    """Tests for OpenAI provider's transcribe() / atranscribe()."""

    def test_sync_transcribe_builds_request_and_maps_response(
        self, provider: OpenAIProvider
    ) -> None:
        from vox import AudioContent

        mock_resp = MagicMock()
        mock_resp.text = "hello world"
        mock_resp.language = "en"
        mock_resp.duration = 1.5

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = MagicMock(return_value=mock_resp)
        provider._sync_client = mock_client

        result = provider.transcribe(
            AudioContent(data="ZmFrZQ==", media_type="audio/wav"),
            model="whisper-1",
            language="en",
            prompt="meeting notes",
        )

        # Provider returns a normalized TranscriptionResponse.
        assert result.text == "hello world"
        assert result.language == "en"
        assert result.duration == 1.5
        assert result.provider == "openai"
        assert result.model == "whisper-1"

        # Request was shaped correctly.
        kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["model"] == "whisper-1"
        assert kwargs["language"] == "en"
        assert kwargs["prompt"] == "meeting notes"
        assert kwargs["response_format"] == "verbose_json"
        # file is a (filename, bytes, mime) tuple — whisper extension
        # is derived from media_type.
        fn, fbytes, mime = kwargs["file"]
        assert fn.endswith(".wav")
        assert mime == "audio/wav"
        assert isinstance(fbytes, bytes)

    def test_sync_transcribe_gpt4o_uses_plain_json(self, provider: OpenAIProvider) -> None:
        from vox import AudioContent

        mock_resp = MagicMock()
        mock_resp.text = "hi"
        # Use spec=[] so that getattr returns None for unknown attrs
        # (otherwise MagicMock would auto-generate them and we'd see
        # a fake "language" / "duration" on the response).
        mock_resp.language = None
        mock_resp.duration = None

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = MagicMock(return_value=mock_resp)
        provider._sync_client = mock_client

        provider.transcribe(
            AudioContent(data="ZmFrZQ=="),
            model="gpt-4o-transcribe",
        )
        kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        # gpt-4o-transcribe doesn't support verbose_json — the request
        # should omit response_format entirely so the API uses its default.
        assert "response_format" not in kwargs

    def test_url_audio_refused(self, provider: OpenAIProvider) -> None:
        from vox import AudioContent
        from vox.errors import InvalidRequestError

        provider._sync_client = MagicMock()
        with pytest.raises(InvalidRequestError, match="URL-sourced AudioContent"):
            provider.transcribe(
                AudioContent(source_type="url", data="https://x/clip.wav"),
                model="whisper-1",
            )

    async def test_async_transcribe_parity(self, provider: OpenAIProvider) -> None:
        from vox import AudioContent

        mock_resp = MagicMock()
        mock_resp.text = "async hi"
        mock_resp.language = "en"
        mock_resp.duration = 0.5

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_resp)
        provider._async_client = mock_client

        result = await provider.atranscribe(
            AudioContent(data="ZmFrZQ=="),
            model="whisper-1",
        )
        assert result.text == "async hi"


class TestOpenAISynthesize:
    """Tests for OpenAI provider's synthesize() / asynthesize()."""

    def test_sync_synthesize_builds_request_and_wraps_response(
        self, provider: OpenAIProvider
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.content = b"\xff\xf3synth-audio-bytes"

        mock_client = MagicMock()
        mock_client.audio.speech.create = MagicMock(return_value=mock_resp)
        provider._sync_client = mock_client

        result = provider.synthesize(
            "Hello world",
            voice="alloy",
            model="tts-1",
            speed=1.25,
        )

        # Request was shaped correctly.
        kwargs = mock_client.audio.speech.create.call_args.kwargs
        assert kwargs["model"] == "tts-1"
        assert kwargs["input"] == "Hello world"
        assert kwargs["voice"] == "alloy"
        assert kwargs["response_format"] == "mp3"
        assert kwargs["speed"] == 1.25

        # Response wrapped as AudioContent with default mp3 MIME.
        from vox import AudioContent

        assert isinstance(result, AudioContent)
        assert result.media_type == "audio/mp3"

    def test_sync_synthesize_honours_format(self, provider: OpenAIProvider) -> None:
        mock_resp = MagicMock()
        mock_resp.content = b"RIFF\x00\x00WAVE"
        mock_client = MagicMock()
        mock_client.audio.speech.create = MagicMock(return_value=mock_resp)
        provider._sync_client = mock_client

        result = provider.synthesize(
            "Hi",
            voice="alloy",
            model="tts-1",
            format="wav",
        )
        kwargs = mock_client.audio.speech.create.call_args.kwargs
        assert kwargs["response_format"] == "wav"
        assert result.media_type == "audio/wav"

    def test_sync_synthesize_passes_instructions_for_gpt4o_mini_tts(
        self, provider: OpenAIProvider
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.content = b"\xff\xf3"
        mock_client = MagicMock()
        mock_client.audio.speech.create = MagicMock(return_value=mock_resp)
        provider._sync_client = mock_client

        provider.synthesize(
            "Hi",
            voice="alloy",
            model="gpt-4o-mini-tts",
            instructions="Speak in a cheerful tone",
        )
        kwargs = mock_client.audio.speech.create.call_args.kwargs
        assert kwargs["instructions"] == "Speak in a cheerful tone"

    async def test_async_synthesize_parity(self, provider: OpenAIProvider) -> None:
        mock_resp = MagicMock()
        mock_resp.content = b"\xff\xf3async-audio"
        mock_client = MagicMock()
        mock_client.audio.speech.create = AsyncMock(return_value=mock_resp)
        provider._async_client = mock_client

        result = await provider.asynthesize("Hi", voice="alloy", model="tts-1")
        from vox import AudioContent

        assert isinstance(result, AudioContent)
