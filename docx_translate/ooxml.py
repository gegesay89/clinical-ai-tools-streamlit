"""OOXML-based DOCX translation while preserving document structure."""

from __future__ import annotations

import html
import io
import re
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Literal

from lxml import etree

from .providers import TranslationProviderError, Translator

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"

W_P = f"{{{W_NS}}}p"
W_T = f"{{{W_NS}}}t"
A_P = f"{{{A_NS}}}p"
A_T = f"{{{A_NS}}}t"
XML_SPACE = f"{{{XML_NS}}}space"

Mode = Literal["paragraph", "runs"]
ProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class DocxTranslationSummary:
    translated_units: int = 0
    skipped_units: int = 0
    translated_characters: int = 0
    translated_parts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _TextGroup:
    part_name: str
    text_nodes: list[etree._Element]
    original_text: str
    mode: Mode


@dataclass(slots=True)
class _XmlPart:
    name: str
    root: etree._Element
    standalone: bool | None


def translate_docx_bytes(
    input_bytes: bytes,
    translator: Translator,
    *,
    source_language: str = "auto",
    target_language: str = "French",
    mode: Mode = "paragraph",
    include_headers_footers: bool = True,
    include_notes_comments: bool = True,
    batch_size: int = 20,
    progress_callback: ProgressCallback | None = None,
) -> tuple[bytes, DocxTranslationSummary]:
    """Translate a DOCX file and return new DOCX bytes plus a summary."""

    if not zipfile.is_zipfile(io.BytesIO(input_bytes)):
        raise ValueError("Uploaded file is not a valid .docx zip package.")

    if mode == "runs" and not getattr(translator, "supports_xml_segments", False):
        mode = "paragraph"

    summary = DocxTranslationSummary()
    parser = etree.XMLParser(resolve_entities=False, remove_blank_text=False, huge_tree=True)
    parsed_parts: dict[str, _XmlPart] = {}
    groups: list[_TextGroup] = []

    with zipfile.ZipFile(io.BytesIO(input_bytes), "r") as zin:
        infos = zin.infolist()
        raw_parts = {info.filename: zin.read(info.filename) for info in infos}

    for name, data in raw_parts.items():
        if not _should_translate_part(
            name,
            include_headers_footers=include_headers_footers,
            include_notes_comments=include_notes_comments,
        ):
            continue
        try:
            tree = etree.fromstring(data, parser=parser)
        except etree.XMLSyntaxError as exc:
            summary.warnings.append(f"Skipped invalid XML part {name}: {exc}")
            continue
        standalone = tree.getroottree().docinfo.standalone
        part = _XmlPart(name=name, root=tree, standalone=standalone)
        parsed_parts[name] = part
        part_groups = _collect_groups(name, tree, mode)
        if part_groups:
            summary.translated_parts.append(name)
            groups.extend(part_groups)

    if not groups:
        summary.warnings.append("No translatable Word text was found.")
        return input_bytes, summary

    total = len(groups)
    translated_count = 0
    for group_batch in _chunk_groups(groups, batch_size=batch_size):
        source_texts = [_source_text_for_group(group) for group in group_batch]
        translations = translator.translate_texts(
            source_texts,
            source_language=source_language,
            target_language=target_language,
        )
        if len(translations) != len(group_batch):
            raise TranslationProviderError(
                f"Provider returned {len(translations)} translations for "
                f"{len(group_batch)} source texts."
            )

        for group, translated in zip(group_batch, translations, strict=True):
            if group.mode == "runs":
                applied = _apply_segmented_translation(group, translated)
                if not applied:
                    summary.warnings.append(
                        f"Fell back to paragraph-level formatting in {group.part_name}."
                    )
                    _apply_paragraph_translation(group, _strip_segment_tags(translated))
            else:
                _apply_paragraph_translation(group, translated)
            translated_count += 1
            summary.translated_units += 1
            summary.translated_characters += len(group.original_text)
            if progress_callback:
                progress_callback(
                    translated_count,
                    total,
                    f"Translated {translated_count} of {total} text blocks",
                )

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        with zipfile.ZipFile(io.BytesIO(input_bytes), "r") as zin:
            for info in zin.infolist():
                part = parsed_parts.get(info.filename)
                data = (
                    _serialize_xml(part)
                    if part is not None
                    else zin.read(info.filename)
                )
                zout.writestr(info, data)

    summary.skipped_units = max(0, len(groups) - summary.translated_units)
    return output.getvalue(), summary


def _should_translate_part(
    name: str,
    *,
    include_headers_footers: bool,
    include_notes_comments: bool,
) -> bool:
    if name == "word/document.xml":
        return True
    if include_headers_footers and re.match(r"word/(header|footer)\d+\.xml$", name):
        return True
    if include_notes_comments and name in {
        "word/footnotes.xml",
        "word/endnotes.xml",
        "word/comments.xml",
    }:
        return True
    if name.startswith("word/charts/") and name.endswith(".xml"):
        return True
    return False


def _collect_groups(part_name: str, root: etree._Element, mode: Mode) -> list[_TextGroup]:
    groups: list[_TextGroup] = []
    seen: set[int] = set()

    for paragraph in root.iter(W_P):
        text_nodes = [node for node in paragraph.iter(W_T) if node.text]
        _append_group(groups, part_name, text_nodes, mode)
        seen.update(id(node) for node in text_nodes)

    for paragraph in root.iter(A_P):
        text_nodes = [
            node for node in paragraph.iter(A_T) if node.text and id(node) not in seen
        ]
        _append_group(groups, part_name, text_nodes, mode)

    return groups


def _append_group(
    groups: list[_TextGroup],
    part_name: str,
    text_nodes: list[etree._Element],
    mode: Mode,
) -> None:
    if not text_nodes:
        return
    original_text = "".join(node.text or "" for node in text_nodes)
    if _should_translate_text(original_text):
        groups.append(
            _TextGroup(
                part_name=part_name,
                text_nodes=text_nodes,
                original_text=original_text,
                mode=mode,
            )
        )


def _should_translate_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if not any(char.isalpha() for char in stripped):
        return False
    if re.fullmatch(r"(https?://|www\.)\S+", stripped, flags=re.I):
        return False
    if re.fullmatch(r"\S+@\S+\.\S+", stripped):
        return False
    return True


def _source_text_for_group(group: _TextGroup) -> str:
    if group.mode != "runs":
        return group.original_text

    segments: list[str] = []
    for idx, node in enumerate(group.text_nodes):
        text = node.text or ""
        segments.append(f'<seg id="{idx}">{html.escape(text)}</seg>')
    return "".join(segments)


def _apply_paragraph_translation(group: _TextGroup, translated: str) -> None:
    target = _dominant_text_node(group.text_nodes)
    for node in group.text_nodes:
        node.text = ""
        node.attrib.pop(XML_SPACE, None)

    target.text = translated
    _set_space_preserve_if_needed(target)


def _dominant_text_node(nodes: list[etree._Element]) -> etree._Element:
    def score(node: etree._Element) -> int:
        text = node.text or ""
        return sum(char.isalpha() for char in text)

    return max(nodes, key=score)


def _apply_segmented_translation(group: _TextGroup, translated: str) -> bool:
    matches = list(re.finditer(r'<seg\s+id="(\d+)">(.*?)</seg>', translated, flags=re.S))
    if len(matches) != len(group.text_nodes):
        return False

    seen_ids: set[int] = set()
    values = [""] * len(group.text_nodes)
    for match in matches:
        idx = int(match.group(1))
        if idx >= len(group.text_nodes) or idx in seen_ids:
            return False
        values[idx] = html.unescape(match.group(2))
        seen_ids.add(idx)

    if seen_ids != set(range(len(group.text_nodes))):
        return False

    for node, value in zip(group.text_nodes, values, strict=True):
        node.text = value
        _set_space_preserve_if_needed(node)
    return True


def _strip_segment_tags(text: str) -> str:
    parts = re.findall(r'<seg\s+id="\d+">(.*?)</seg>', text, flags=re.S)
    if parts:
        return "".join(html.unescape(part) for part in parts)
    return re.sub(r"</?seg[^>]*>", "", text)


def _set_space_preserve_if_needed(node: etree._Element) -> None:
    text = node.text or ""
    if text[:1].isspace() or text[-1:].isspace():
        node.set(XML_SPACE, "preserve")
    else:
        node.attrib.pop(XML_SPACE, None)


def _chunk_groups(groups: list[_TextGroup], *, batch_size: int) -> list[list[_TextGroup]]:
    batch_size = max(1, batch_size)
    batches: list[list[_TextGroup]] = []
    current: list[_TextGroup] = []
    current_chars = 0

    for group in groups:
        source_len = len(_source_text_for_group(group))
        if current and (len(current) >= batch_size or current_chars + source_len > 12000):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(group)
        current_chars += source_len
    if current:
        batches.append(current)
    return batches


def _serialize_xml(part: _XmlPart) -> bytes:
    kwargs = {
        "encoding": "UTF-8",
        "xml_declaration": True,
    }
    if part.standalone is not None:
        kwargs["standalone"] = part.standalone
    return etree.tostring(part.root, **kwargs)
