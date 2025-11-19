import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


def _find_heading_indices(lines: List[str]) -> List[Tuple[int, str, str, str]]:
    indices: List[Tuple[int, str, str, str]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        heading_match = re.match(r"^#{1,4}\s+\*\*(.+)\*\*$", stripped)
        if not heading_match:
            continue
        content = heading_match.group(1).strip()
        inner_match = re.match(
            r"(?P<section>\d+\.\d+\.\d+)(?:\.[A-Z]+)?\.\s*(?P<pid>[A-Z]+\.\d+)\s+(?P<name>.+)",
            content,
        )
        if not inner_match:
            continue
        section = inner_match.group("section")
        pid = inner_match.group("pid")
        name = _normalize(inner_match.group("name"))
        section_heading = f"{section} {pid} {name}"
        indices.append((idx, pid, name, section_heading))
    return indices


def _matches_heading(line: str, keyword: str) -> bool:
    stripped = line.strip()
    return stripped.endswith(keyword) and stripped.startswith("#")


def _extract_paragraph(block: List[str], keyword: str) -> str:
    try:
        start = next(i for i, line in enumerate(block) if _matches_heading(line, keyword))
    except StopIteration:
        return ""
    collected: List[str] = []
    for line in block[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            if collected:
                break
            continue
        if stripped.startswith("#") or stripped.startswith("####"):
            break
        if stripped.startswith("|") or stripped.startswith("![]") or "©" in stripped:
            continue
        collected.append(stripped)
    return _normalize(" ".join(collected))


def _extract_outcomes(block: List[str]) -> List[str]:
    outcomes: List[str] = []
    try:
        start = next(i for i, line in enumerate(block) if _matches_heading(line, "**Process outcomes**"))
    except StopIteration:
        return outcomes
    current = ""
    for line in block[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("####"):
            break
        if stripped.startswith("|") or stripped.startswith("![]") or "©" in stripped:
            continue
        if stripped.startswith("- "):
            if current:
                outcomes.append(_normalize(current))
            current = stripped[2:].strip()
        elif re.match(r"^\d\)", stripped):
            if current:
                outcomes.append(_normalize(current))
            current = stripped
        else:
            if current:
                current += " " + stripped
            else:
                current = stripped
    if current:
        outcomes.append(_normalize(current))
    return outcomes


def _extract_base_practices(block: List[str], process_id: str) -> List[Dict[str, object]]:
    start = None
    for i, line in enumerate(block):
        stripped = line.strip()
        if stripped == "# **Base Practices**":
            start = i + 1
            break
        if re.search(rf"\*\*{re.escape(process_id)}\.BP\d+", stripped):
            start = i
            break
    if start is None:
        return []
    base_practices: List[Dict[str, object]] = []
    current: Dict[str, object] | None = None
    desc_parts: List[str] = []
    bp_pattern = re.compile(rf"\*\*({re.escape(process_id)}\.BP\d+):\s*(.+?)\.\*\*\s*(.*)")
    note_pattern = re.compile(r"\*?Note\s*\d+.*", re.IGNORECASE)

    for line in block[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("####"):
            break
        if stripped.startswith("|") or stripped.startswith("![]") or "©" in stripped:
            continue

        candidate = stripped.lstrip("- ").lstrip()
        bp_match = bp_pattern.match(candidate)
        if bp_match:
            if current:
                current["description"] = _normalize(" ".join(desc_parts))
                base_practices.append(current)
            bp_id, title, remainder = bp_match.groups()
            desc_parts = []
            if remainder:
                desc_parts.append(remainder)
            current = {
                "id": bp_id,
                "title": title.strip(),
                "description": "",
                "notes": [],
            }
            continue

        if current is None:
            continue

        note_match = note_pattern.match(candidate.strip("*"))
        if note_match:
            note_text = candidate.strip("* ").rstrip("*")
            current["notes"].append(_normalize(note_text))
            continue

        desc_parts.append(stripped)

    if current:
        current["description"] = _normalize(" ".join(desc_parts))
        base_practices.append(current)
    return base_practices


def chunk_aspice_pam(file_path: str) -> List[Dict[str, object]]:
    text = Path(file_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    headings = _find_heading_indices(lines)
    if not headings:
        raise ValueError("No ASPICE process headings found in source document.")

    entries: List[Dict[str, object]] = []
    for idx, (start_idx, proc_id, proc_name, section_heading) in enumerate(headings):
        end_idx = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        block = lines[start_idx:end_idx]
        purpose = _extract_paragraph(block, "**Process purpose**")
        outcomes = _extract_outcomes(block)
        base_practices = _extract_base_practices(block, proc_id)
        if not base_practices:
            raise ValueError(f"No base practices parsed for {proc_id}")

        entry = {
            "process_id": proc_id,
            "process_name": proc_name,
            "process_purpose": purpose,
            "process_outcomes": outcomes,
            "base_practices": base_practices,
            "source": {
                "document": file_path,
                "section_heading": section_heading,
            },
            "vector_index": f"aspice_normative_{proc_id.lower().replace('.', '_')}",
        }
        entries.append(entry)
    return entries


def main() -> None:
    source_markdown = Path("corpus/aspice/Automotive-SPICE-PAM-v40/Automotive-SPICE-PAM-v40-cleaned.md")
    chunks = chunk_aspice_pam(str(source_markdown))
    output_path = Path("aspice_chunks.json")
    output_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(chunks)} chunk(s) to {output_path}")


if __name__ == "__main__":
    main()
