from __future__ import annotations

import pytest

from docx_translate import (
    BedrockOpenAITranslator,
    TranslationProviderError,
    bedrock_translation_enabled,
)


def test_bedrock_translation_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BEDROCK_TRANSLATION_ENABLED", raising=False)

    assert bedrock_translation_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_bedrock_translation_can_be_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("BEDROCK_TRANSLATION_ENABLED", value)

    assert bedrock_translation_enabled() is True


def test_disabled_bedrock_translator_stops_before_an_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BEDROCK_TRANSLATION_ENABLED", "false")
    translator = BedrockOpenAITranslator(
        model_id="example-model",
        region_name="us-east-2",
    )

    with pytest.raises(TranslationProviderError, match="temporarily disabled"):
        translator.translate_texts(
            ["Clinical note"],
            source_language="English",
            target_language="French",
        )
