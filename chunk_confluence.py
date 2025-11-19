import json
import re
from pathlib import Path
from typing import Dict, List

HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _derive_page_metadata(md_path: Path, lines: List[str]) -> Dict[str, str]:
    stem = md_path.stem
    if "_" in stem:
        slug, page_id = stem.rsplit("_", 1)
    else:
        slug, page_id = stem, stem
    page_title = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            candidate = stripped.lstrip("#").strip()
            if candidate:
                page_title = candidate
                break
    if not page_title:
        page_title = slug.replace("-", " ").strip() or stem
    return {"page_id": page_id, "page_title": page_title}


def _finalize_chunk(
    chunks: List[Dict[str, object]],
    page_id: str,
    heading_path: List[str],
    content_lines: List[str],
    start_line: int | None,
    end_line: int | None,
    document: str,
) -> None:
    if not content_lines:
        return
    text = "\n".join(content_lines).strip()
    if not text:
        return
    chunk = {
        "chunk_id": f"{page_id}::{len(chunks) + 1}",
        "page_id": page_id,
        "page_title": heading_path[0] if heading_path else "",
        "heading_path": heading_path,
        "content": text,
        "source": {
            "document": document,
            "start_line": start_line,
            "end_line": end_line,
        },
    }
    chunks.append(chunk)


def _chunk_single_markdown(md_path: Path) -> List[Dict[str, object]]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    metadata = _derive_page_metadata(md_path, lines)
    page_id = metadata["page_id"]
    page_title = metadata["page_title"]

    heading_stack: List[tuple[int, str]] = []
    heading_path: List[str] = [page_title]
    content_lines: List[str] = []
    start_line: int | None = None
    end_line: int | None = None
    chunks: List[Dict[str, object]] = []

    for idx, raw_line in enumerate(lines, start=1):
        stripped_leading = raw_line.lstrip()
        heading_match = HEADING_REGEX.match(stripped_leading)
        if heading_match:
            title = heading_match.group(2).strip()
            if not title:
                continue
            level = len(heading_match.group(1))
            _finalize_chunk(
                chunks,
                page_id,
                heading_path.copy(),
                content_lines,
                start_line,
                end_line,
                str(md_path),
            )
            content_lines = []
            start_line = None
            end_line = None
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            heading_path = [title for _, title in heading_stack]
            if heading_stack[0][1] != page_title:
                heading_path = [page_title] + heading_path
            continue

        if not heading_stack:
            heading_stack.append((1, page_title))
            heading_path = [page_title]

        if start_line is None and stripped_leading.strip():
            start_line = idx
        if stripped_leading.strip():
            end_line = idx
        content_lines.append(raw_line)

    _finalize_chunk(
        chunks,
        page_id,
        heading_path.copy(),
        content_lines,
        start_line,
        end_line,
        str(md_path),
    )
    return chunks


def chunk_confluence_markdown(source_dir: str, output_path: str) -> None:
    md_dir = Path(source_dir)
    if not md_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {source_dir}")

    all_chunks: List[Dict[str, object]] = []
    for md_path in sorted(md_dir.glob("*.md")):
        file_chunks = _chunk_single_markdown(md_path)
        all_chunks.extend(file_chunks)

    Path(output_path).write_text(json.dumps(all_chunks, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(all_chunks)} chunk(s) from {len(list(md_dir.glob('*.md')))} Markdown files to {output_path}"
    )


def main() -> None:
    chunk_confluence_markdown("corpus/SWT1AQ_md", "confluence_chunks.json")


if __name__ == "__main__":
    main()
