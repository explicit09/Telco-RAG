"""Deterministic, corpus-independent document ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from .models import Chunk

_WORD = re.compile(r"\S+")
_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _chunk_id(corpus: str, document: str, section: str, text: str) -> str:
    value = "\x1f".join((corpus, document, section, text))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    words = _WORD.findall(text)
    result: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if current and len(candidate) > max_chars:
            result.append(current)
            current = word
        else:
            current = candidate
    if current:
        result.append(current)
    return result or [text[:max_chars]]


def _plain_blocks(path: Path) -> list[tuple[str, str, str, int, dict]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    blocks: list[tuple[str, str, str, int, dict]] = []
    heading = ""
    paragraph: list[str] = []

    def flush():
        if paragraph:
            ordinal = len(blocks) + 1
            blocks.append((" ".join(paragraph), heading, path.name, ordinal,
                           {"source_kind": "paragraph", "ordinal": ordinal}))
            paragraph.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line) if path.suffix.lower() == ".md" else None
        if match:
            flush()
            heading = match.group(1).strip()
        else:
            paragraph.append(line)
    flush()
    return blocks


W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
def _word_inline_text(node, unsupported=None):
    unsupported = set() if unsupported is None else unsupported
    tag=node.tag.rsplit('}',1)[-1]
    math=node.tag.startswith('{'+M+'}')
    def child(name):
        item=node.find('{'+M+'}'+name)
        return _word_inline_text(item,unsupported) if item is not None else ''
    def prop(group,name,default):
        item=node.find('{'+M+'}'+group+'/{'+M+'}'+name)
        return item.get('{'+M+'}val',default) if item is not None else default
    if tag=='t' and node.tag in ('{'+W+'}t','{'+M+'}t'):return node.text or ''
    if not math and tag in ('tab','br'):return ' '
    if tag.endswith('Pr'):return ''
    if math:
        if tag=='sSub':return '('+child('e')+')_{'+child('sub')+'}'
        if tag=='sSup':return '('+child('e')+')^{'+child('sup')+'}'
        if tag=='sSubSup':return '('+child('e')+')_{'+child('sub')+'}^{'+child('sup')+'}'
        if tag=='f':return '(('+child('num')+')/('+child('den')+'))'
        if tag=='d':return prop('dPr','begChr','(')+prop('dPr','sepChr','|').join(_word_inline_text(x,unsupported) for x in node.findall('{'+M+'}e'))+prop('dPr','endChr',')')
        if tag=='func':return child('fName')+'('+child('e')+')'
        if tag=='nary':
            symbol=prop('naryPr','chr','[unspecified n-ary operator]')
            if symbol.startswith('['):unsupported.add('nary operator without explicit chr')
            return symbol+'_{'+child('sub')+'}^{'+child('sup')+'}('+child('e')+')'
        if tag=='acc':return 'accent['+prop('accPr','chr','unspecified')+']('+child('e')+')'
        if tag=='bar':return 'bar['+prop('barPr','pos','unspecified')+']('+child('e')+')'
        if tag=='limLow':return child('e')+'_{'+child('lim')+'}'
        if tag not in {'oMath','oMathPara','r','e','sub','sup','num','den','fName','lim'}:
            unsupported.add(tag)
            return '[unsupported '+tag+': '+''.join(_word_inline_text(x,unsupported) for x in node)+']'
    return ''.join(_word_inline_text(x,unsupported) for x in node)


def _docx_blocks(path: Path) -> list[tuple[str, str, str, int, dict]]:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except (OSError, KeyError, zipfile.BadZipFile, UnicodeError) as exc:
        raise ValueError(f"malformed DOCX {path}: {exc}") from exc
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ValueError(f"malformed DOCX XML {path}: {exc}") from exc

    body = root.find("w:body", _NS)
    if body is None:
        raise ValueError(f"malformed DOCX {path}: missing document body")
    blocks: list[tuple[str, str, str, int, dict]] = []
    heading = ""
    ordinal = 0
    table_ordinal = 0
    for child in body:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            value = _word_inline_text(child).strip()
            if not value:
                continue
            style = child.find("w:pPr/w:pStyle", _NS)
            style_name = style.get("{" + _NS["w"] + "}val", "") if style is not None else ""
            if style_name.lower().startswith("heading"):
                heading = value
                continue
            ordinal += 1
            blocks.append((value, heading, path.name, ordinal, {"source_kind": "paragraph", "ordinal": ordinal}))
        elif tag == "tbl":
            table_ordinal += 1
            rows: list[str] = []
            for row in child.findall("w:tr", _NS):
                cells = [_word_inline_text(cell).strip() for cell in row.findall("w:tc", _NS)]
                row_text = " | ".join(cells)
                if any(cells):
                    rows.append(row_text)
            if rows:
                header = rows[0]
                for row_index, row in enumerate(rows):
                    ordinal += 1
                    value = row if row_index == 0 else f"{header} | {row}"
                    blocks.append((value, heading, path.name, ordinal, {"source_kind": "table", "ordinal": ordinal, "table_ordinal": table_ordinal, "row": row_index}))
    return blocks


def _jsonl_blocks(path: Path) -> list[tuple[str, str, str, int, dict, dict]]:
    blocks = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    for ordinal, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL at line {ordinal}: {exc}") from exc
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError(f"JSONL line {ordinal} must be an object with non-empty text")
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"JSONL line {ordinal} metadata must be an object")
        blocks.append((item["text"].strip(), str(item.get("section", "")), str(item.get("source", path.name)), ordinal, {"source_kind": "jsonl", "ordinal": ordinal}, item))
    return blocks


def ingest_file(path: Path, *, corpus_id: str, document_id: str | None = None, release: str = "", max_chars: int = 1800) -> list[Chunk]:
    """Read a supported file into stable chunks; malformed input raises ValueError."""
    path = Path(path)
    if not corpus_id or max_chars <= 0:
        raise ValueError("corpus_id must be non-empty and max_chars must be positive")
    document = document_id or path.stem
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        raw = [(a, b, c, d, e, {}) for a, b, c, d, e in _plain_blocks(path)]
    elif suffix == ".docx":
        raw = [(a, b, c, d, e, {}) for a, b, c, d, e in _docx_blocks(path)]
    elif suffix == ".jsonl":
        raw = _jsonl_blocks(path)
    else:
        raise ValueError(f"unsupported file type: {path.suffix or '<none>'}")
    chunks: list[Chunk] = []
    for text, section, source, ordinal, metadata, item in raw:
        context = f"{section}\n{text}" if section else text
        for piece_index, piece in enumerate(_split(context, max_chars)):
            merged = dict(metadata)
            if item:
                merged.update({"metadata": item.get("metadata", {}), "title": item.get("title", ""), "normalized_release": item.get("release", "")})
            chunks.append(Chunk(id=_chunk_id(corpus_id, document, section, f"{ordinal}:{piece_index}:{piece}"), corpus_id=corpus_id, document_id=document, source=source, title=str(item.get("title", "")) if item else path.stem, section=section, text=piece, release=str(item.get("release", release)) if item else release, metadata=merged))
    if not chunks:
        raise ValueError("document contains no ingestible text")
    return chunks
