"""Translation provider adapters.

The app keeps provider-specific code here so the DOCX rewrite layer can stay
focused on preserving the Word package.
"""

from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import requests


class TranslationProviderError(RuntimeError):
    """Raised when a translation provider cannot complete a request."""


class Translator(Protocol):
    name: str
    supports_xml_segments: bool

    def translate_texts(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
    ) -> list[str]:
        """Translate a batch of strings."""


def build_translator(
    provider: str,
    *,
    openai_model: str,
    openai_api_key: str | None = None,
    deepl_api_key: str | None = None,
    aws_profile: str | None = None,
    aws_region: str | None = None,
    bedrock_model_id: str | None = None,
) -> Translator:
    """Create a translator from UI/environment configuration."""

    provider = provider.lower()
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    deepl_api_key = deepl_api_key or os.getenv("DEEPL_API_KEY")

    if provider == "auto":
        if _local_marian_is_available():
            return LocalMarianTranslator()
        if openai_api_key:
            return OpenAITranslator(api_key=openai_api_key, model=openai_model)
        if deepl_api_key:
            return DeepLTranslator(api_key=deepl_api_key)
        return DemoTranslator()
    if provider == "local_marian":
        return LocalMarianTranslator()
    if provider == "bedrock_openai":
        return BedrockOpenAITranslator(
            profile_name=aws_profile or os.getenv("AWS_PROFILE") or None,
            region_name=aws_region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-2",
            model_id=bedrock_model_id
            or os.getenv("BEDROCK_OPENAI_MODEL_ID")
            or "openai.gpt-5.5",
        )
    if provider == "argos":
        return ArgosTranslator()
    if provider == "openai":
        if not openai_api_key:
            raise TranslationProviderError("OPENAI_API_KEY is not set.")
        return OpenAITranslator(api_key=openai_api_key, model=openai_model)
    if provider == "deepl":
        if not deepl_api_key:
            raise TranslationProviderError("DEEPL_API_KEY is not set.")
        return DeepLTranslator(api_key=deepl_api_key)
    if provider == "demo":
        return DemoTranslator()
    raise TranslationProviderError(f"Unknown translation provider: {provider}")


@dataclass(slots=True)
class BedrockOpenAITranslator:
    """Translator through Amazon Bedrock.

    Uses AWS credentials/profile locally. OpenAI GPT-5.x Bedrock models use
    Bedrock's OpenAI-compatible Responses endpoint; regular foundation models
    and inference profiles use Bedrock Runtime Converse.
    """

    model_id: str
    region_name: str
    profile_name: str | None = None
    name: str = "Amazon Bedrock"
    supports_xml_segments: bool = True
    max_tokens: int = 4096
    temperature: float = 0.0

    def translate_texts(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
    ) -> list[str]:
        if not texts:
            return []

        if _uses_bedrock_openai_responses(self.model_id):
            raw = self._call_bedrock_openai_responses(
                texts,
                source_language=source_language,
                target_language=target_language,
            )
        else:
            raw = self._call_bedrock_runtime_converse(
                texts,
                source_language=source_language,
                target_language=target_language,
            )

        return _parse_translation_json(raw, expected_count=len(texts))

    def _call_bedrock_openai_responses(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
    ) -> str:
        try:
            from openai import BedrockOpenAI
        except ImportError as exc:
            raise TranslationProviderError(
                "The openai package with Bedrock support is not installed. "
                "Run `pip install openai`."
            ) from exc

        client_kwargs: dict[str, str] = {"aws_region": self.region_name}
        if self.profile_name:
            client_kwargs["aws_profile"] = self.profile_name

        bedrock_api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("BEDROCK_API_KEY")
        if bedrock_api_key:
            client_kwargs["api_key"] = bedrock_api_key

        client = BedrockOpenAI(**client_kwargs)
        response = client.responses.create(
            model=self.model_id,
            instructions=self._system_prompt(source_language, target_language),
            input=json.dumps({"texts": texts}, ensure_ascii=False),
            max_output_tokens=self.max_tokens,
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
        return _extract_response_text(response)

    def _call_bedrock_runtime_converse(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
    ) -> str:
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as exc:
            raise TranslationProviderError(
                "boto3 is not installed. Run `pip install boto3 botocore`."
            ) from exc

        session_kwargs = {}
        if self.profile_name:
            session_kwargs["profile_name"] = self.profile_name
        if self.region_name:
            session_kwargs["region_name"] = self.region_name

        try:
            session = boto3.Session(**session_kwargs)
            client = session.client("bedrock-runtime")
            return self._call_converse(
                client,
                texts,
                source_language=source_language,
                target_language=target_language,
            )
        except (BotoCoreError, ClientError) as exc:
            raise TranslationProviderError(f"Bedrock request failed: {exc}") from exc

    def _call_converse(
        self,
        client: object,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
    ) -> str:
        user_payload = json.dumps({"texts": texts}, ensure_ascii=False)

        response = client.converse(
            modelId=self.model_id,
            system=[{"text": self._system_prompt(source_language, target_language)}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_payload}],
                }
            ],
            inferenceConfig={"maxTokens": self.max_tokens},
        )
        chunks: list[str] = []
        for part in (
            response.get("output", {})
            .get("message", {})
            .get("content", [])
        ):
            text = part.get("text")
            if text:
                chunks.append(text)
        if not chunks:
            raise TranslationProviderError("Bedrock returned no text output.")
        return "\n".join(chunks)

    def _system_prompt(self, source_language: str, target_language: str) -> str:
        source_instruction = ""
        if source_language and source_language.lower() not in {"auto", "detect"}:
            source_instruction = f"The source language is {source_language}. "
        return (
            "You are a professional medical document translator. "
            f"{source_instruction}Translate the provided texts into {target_language}. "
            "Preserve medical meaning precisely, including diagnoses, anatomy, "
            "medications, dosages, units, abbreviations, tables, references, numbers, "
            "URLs, email addresses, placeholders, and XML-like tags such as "
            "<seg id=\"0\"> exactly. Do not summarize or simplify. "
            "Do not add commentary. Return only valid JSON with this exact shape: "
            "{\"translations\": [\"...\"]}."
        )


@dataclass(slots=True)
class LocalMarianTranslator:
    """Local Hugging Face MarianMT translator.

    This is a conventional machine-translation model, not a general LLM. The
    first run downloads model weights; later runs use the local cache.
    """

    model_name: str = ""
    name: str = "Local MarianMT"
    supports_xml_segments: bool = False
    batch_size: int = 8

    def translate_texts(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
    ) -> list[str]:
        if not texts:
            return []

        from_code = _local_language_code(source_language, default="en")
        to_code = _local_language_code(target_language, default="fr")
        model_name = self.model_name or f"Helsinki-NLP/opus-mt-{from_code}-{to_code}"
        device = os.getenv("LOCAL_TRANSLATION_DEVICE", "cpu")
        tokenizer, model, torch, device = _load_marian_model(model_name, device)

        outputs: list[str] = []
        for batch in _chunk_texts(texts, max(1, self.batch_size)):
            translated_batch = self._translate_batch(batch, tokenizer, model, torch, device)
            outputs.extend(translated_batch)
        return outputs

    def _translate_batch(
        self,
        texts: list[str],
        tokenizer: object,
        model: object,
        torch: object,
        device: str,
    ) -> list[str]:
        output_by_index: dict[int, str] = {}
        to_translate: list[str] = []
        index_map: list[int] = []

        for idx, text in enumerate(texts):
            if _has_real_text(text):
                to_translate.append(text)
                index_map.append(idx)
            else:
                output_by_index[idx] = text

        if to_translate:
            inputs = tokenizer(
                to_translate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=512)
            translated = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for idx, value in zip(index_map, translated, strict=True):
                output_by_index[idx] = value

        return [output_by_index[idx] for idx in range(len(texts))]


@dataclass(slots=True)
class ArgosTranslator:
    """Offline translator backed by Argos Translate.

    The first English-to-French run may download and install the local model.
    After that, translation is local and does not need an API key.
    """

    name: str = "Argos Translate"
    supports_xml_segments: bool = False
    auto_install_models: bool = True

    def translate_texts(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
    ) -> list[str]:
        if not texts:
            return []

        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError as exc:
            raise TranslationProviderError(
                "Argos Translate is not installed. Run `pip install argostranslate`."
            ) from exc

        from_code = _argos_language_code(source_language, default="en")
        to_code = _argos_language_code(target_language, default="fr")
        self._ensure_model(argostranslate.package, from_code, to_code)

        return [
            argostranslate.translate.translate(text, from_code, to_code)
            if _has_real_text(text)
            else text
            for text in texts
        ]

    def _ensure_model(self, package_module: object, from_code: str, to_code: str) -> None:
        if _argos_pair_installed(from_code, to_code):
            return
        if not self.auto_install_models:
            raise TranslationProviderError(
                f"Argos model {from_code}->{to_code} is not installed."
            )

        package_module.update_package_index()
        available_packages = package_module.get_available_packages()
        package = next(
            (
                item
                for item in available_packages
                if item.from_code == from_code and item.to_code == to_code
            ),
            None,
        )
        if package is None:
            raise TranslationProviderError(
                f"No Argos model is available for {from_code}->{to_code}."
            )

        package_path = package.download()
        package_module.install_from_path(package_path)
        if not _argos_pair_installed(from_code, to_code):
            raise TranslationProviderError(
                f"Argos model {from_code}->{to_code} was installed but is not available."
            )


@dataclass(slots=True)
class DemoTranslator:
    """A tiny offline translator used for smoke tests and UI demos.

    This is intentionally limited. Real documents should use OpenAI or DeepL.
    """

    name: str = "Demo translator"
    supports_xml_segments: bool = True

    _dictionary = {
        "hello": "bonjour",
        "world": "monde",
        "patient": "patient",
        "patients": "patients",
        "document": "document",
        "summary": "resume",
        "this": "ceci",
        "is": "est",
        "a": "un",
        "test": "test",
        "table": "tableau",
        "header": "en-tete",
        "footer": "pied de page",
        "name": "nom",
        "date": "date",
        "status": "statut",
        "approved": "approuve",
        "pending": "en attente",
    }

    def translate_texts(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
    ) -> list[str]:
        return [self._translate_one(text) for text in texts]

    def _translate_one(self, text: str) -> str:
        def replace_segment(match: re.Match[str]) -> str:
            attrs = match.group(1)
            content = html.unescape(match.group(2))
            return f"<seg{attrs}>{html.escape(self._translate_plain(content))}</seg>"

        if "<seg" in text:
            return re.sub(r"<seg([^>]*)>(.*?)</seg>", replace_segment, text, flags=re.S)
        return self._translate_plain(text)

    def _translate_plain(self, text: str) -> str:
        words = re.split(r"(\W+)", text)
        translated: list[str] = []
        for word in words:
            replacement = self._dictionary.get(word.lower())
            if replacement and word[:1].isupper():
                replacement = replacement[:1].upper() + replacement[1:]
            translated.append(replacement or word)
        output = "".join(translated).strip()
        return output or text


@dataclass(slots=True)
class DeepLTranslator:
    """DeepL REST API translator."""

    api_key: str
    name: str = "DeepL"
    supports_xml_segments: bool = False
    timeout_seconds: int = 90

    def translate_texts(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
    ) -> list[str]:
        if not texts:
            return []

        target_lang = _deepl_language_code(target_language)
        payload: list[tuple[str, str]] = [
            ("auth_key", self.api_key),
            ("target_lang", target_lang),
        ]
        if source_language and source_language.lower() not in {"auto", "detect"}:
            payload.append(("source_lang", source_language.upper()))
        payload.extend(("text", text) for text in texts)

        url = os.getenv("DEEPL_API_URL") or (
            "https://api-free.deepl.com/v2/translate"
            if self.api_key.endswith(":fx")
            else "https://api.deepl.com/v2/translate"
        )

        response = requests.post(url, data=payload, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise TranslationProviderError(
                f"DeepL request failed with HTTP {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        translations = [item["text"] for item in data.get("translations", [])]
        if len(translations) != len(texts):
            raise TranslationProviderError(
                f"DeepL returned {len(translations)} translations for {len(texts)} texts."
            )
        return translations


@dataclass(slots=True)
class OpenAITranslator:
    """OpenAI SDK translator."""

    api_key: str
    model: str
    name: str = "OpenAI"
    supports_xml_segments: bool = True
    timeout_seconds: int = 120

    def translate_texts(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
    ) -> list[str]:
        if not texts:
            return []

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise TranslationProviderError(
                "The openai package is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
        system_prompt = (
            "You are a professional document translator. Translate the provided "
            f"texts into {target_language}. Preserve meaning, numbers, URLs, email "
            "addresses, placeholders, and XML-like tags such as <seg id=\"0\">. "
            "Return only valid JSON with this exact shape: "
            "{\"translations\": [\"...\"]}."
        )
        if source_language and source_language.lower() not in {"auto", "detect"}:
            system_prompt += f" The source language is {source_language}."
        user_payload = json.dumps({"texts": texts}, ensure_ascii=False)

        raw = self._call_openai(client, system_prompt, user_payload)
        return _parse_translation_json(raw, expected_count=len(texts))

    def _call_openai(self, client: object, system_prompt: str, user_payload: str) -> str:
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
            )
            output_text = getattr(response, "output_text", None)
            if output_text:
                return output_text
            return _extract_response_text(response)

        # Fallback for older SDKs.
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""


def _deepl_language_code(target_language: str) -> str:
    normalized = target_language.strip().lower()
    if normalized in {"french", "francais", "fr"}:
        return "FR"
    if len(normalized) in {2, 5}:
        return normalized.upper()
    return "FR"


def _argos_is_available() -> bool:
    try:
        import argostranslate  # noqa: F401
    except ImportError:
        return False
    return True


def _local_marian_is_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _uses_bedrock_openai_responses(model_id: str) -> bool:
    return model_id.startswith("openai.") and not model_id.endswith(":0")


@lru_cache(maxsize=4)
def _load_marian_model(model_name: str, device: str) -> tuple[object, object, object, str]:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise TranslationProviderError(
            "Local MarianMT dependencies are not installed. "
            "Run `pip install transformers sentencepiece torch`."
        ) from exc

    if device == "auto":
        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    if device not in {"cpu", "mps", "cuda"}:
        device = "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as exc:  # noqa: BLE001 - model-load errors should be user-visible.
        raise TranslationProviderError(
            f"Could not load local translation model {model_name!r}: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    return tokenizer, model, torch, device


def _local_language_code(language: str, *, default: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized in {"", "auto", "detect"}:
        return default
    aliases = {
        "english": "en",
        "eng": "en",
        "en-us": "en",
        "en-gb": "en",
        "french": "fr",
        "francais": "fr",
        "français": "fr",
    }
    return aliases.get(normalized, normalized[:2])


def _chunk_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    return [texts[index : index + batch_size] for index in range(0, len(texts), batch_size)]


def _argos_language_code(language: str, *, default: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized in {"", "auto", "detect"}:
        return default
    aliases = {
        "english": "en",
        "eng": "en",
        "en-us": "en",
        "en-gb": "en",
        "french": "fr",
        "francais": "fr",
        "français": "fr",
    }
    return aliases.get(normalized, normalized[:2])


def _argos_pair_installed(from_code: str, to_code: str) -> bool:
    try:
        import argostranslate.translate
    except ImportError:
        return False

    installed_languages = argostranslate.translate.get_installed_languages()
    from_language = next(
        (language for language in installed_languages if language.code == from_code),
        None,
    )
    if from_language is None:
        return False
    return any(translation.to_lang.code == to_code for translation in from_language.translations)


def _has_real_text(text: str) -> bool:
    return bool(text and any(char.isalpha() for char in text))


def _parse_translation_json(raw: str, *, expected_count: int) -> list[str]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            if expected_count == 1 and raw:
                return [_strip_markdown_fence(raw)]
            raise TranslationProviderError("Translation provider did not return JSON.")
        data = json.loads(match.group(0))

    translations = data.get("translations")
    if not isinstance(translations, list) or not all(
        isinstance(item, str) for item in translations
    ):
        if expected_count == 1 and isinstance(data.get("translation"), str):
            return [data["translation"]]
        raise TranslationProviderError("Translation provider returned JSON without translations[].")
    if len(translations) != expected_count:
        raise TranslationProviderError(
            f"Translation provider returned {len(translations)} translations for {expected_count} texts."
        )
    return translations


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    fence_match = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", stripped, flags=re.S)
    return fence_match.group(1).strip() if fence_match else stripped


def _extract_response_text(response: object) -> str:
    """Best-effort extraction for SDK response objects."""

    output = getattr(response, "output", None)
    if output:
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None) or []
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)
    return str(response)
