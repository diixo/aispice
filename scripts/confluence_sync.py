"""
Confluence Stage 1 CLI — Offline pipeline

Implements an offline-first workflow to Evaluate → Pull → Convert/Chunk → Index → Activate
for a Confluence space export already present on disk. This is the Stage 1 script referenced
in api_confluence.md.

Commands (remote first; local mode available for reference)
- evaluate: remote (preferred) — traverse Confluence from a root page id using PAT/basic auth; compute page count and delta vs last state. Local mode scans a local HTML/MD export instead.
- pull: remote (preferred) — fetch page HTML (storage) for added/changed pages into staging; local mode copies local files.
- build: convert HTML→MD when needed, then chunk Markdown to confluence_chunks.staging.json
- activate: atomically activate the staging dataset and rebuild the `confluence` FAISS index

Examples (remote)
- python scripts/confluence_sync.py evaluate --space SWT1AQ --api-root https://your-domain.atlassian.net/wiki/rest/api --root-page-id 12345 --email you@example.com --token YOUR_API_TOKEN
- python scripts/confluence_sync.py pull --space SWT1AQ --api-root https://your-domain.atlassian.net/wiki/rest/api --email you@example.com --token YOUR_API_TOKEN
- python scripts/confluence_sync.py build --space SWT1AQ
- python scripts/confluence_sync.py activate --space SWT1AQ

Local mode (reference)
- python scripts/confluence_sync.py evaluate --space SWT1AQ --source-html corpus/SWT1AQ
- python scripts/confluence_sync.py pull --space SWT1AQ  # copies evaluated local source to staging

Notes
- This script does not fetch from remote Confluence in Stage 1. Remote PAT-based
  integration is planned for Stage 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Optional imports (live in project root)
try:  # noqa: SIM105
    from chunk_confluence import chunk_confluence_markdown  # type: ignore
except Exception:  # pragma: no cover - runtime presence guaranteed in repo
    chunk_confluence_markdown = None  # type: ignore

try:  # noqa: SIM105
    from convert_html_to_md import convert_html_to_md  # type: ignore
except Exception:  # pragma: no cover
    convert_html_to_md = None  # type: ignore

try:  # noqa: SIM105
    from atlassian import Confluence  # type: ignore
except Exception:  # pragma: no cover
    Confluence = None  # type: ignore


@dataclass
class SourceSpec:
    kind: str  # "html" or "md"
    path: str


@dataclass
class Plan:
    space: str
    source: SourceSpec
    pages: int
    files: int
    estimated_seconds: float
    added: List[str]
    changed: List[str]
    deleted: List[str]


class SyncPaths:
    def __init__(self, space: str) -> None:
        self.space = space
        self.root = ROOT
        self.corpus = self.root / "corpus"
        self.state_dir = self.corpus / space / ".sync"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Staging/active roots (mirrors)
        self.staging_html = self.corpus / f"{space}_staging_html"
        self.staging_md = self.corpus / f"{space}_staging_md"
        self.active_md_link = self.corpus / f"{space}_md_active"
        # Chunk outputs
        self.staging_chunks = self.root / "confluence_chunks.staging.json"
        self.active_chunks = self.root / "confluence_chunks.json"
        # State files
        self.plan_path = self.state_dir / "plan.json"
        self.delta_path = self.state_dir / "delta.json"
        self.state_path = self.state_dir / "state.json"
        # Perf log
        self.perf_log = self.root / "logs" / "confluence_perf.jsonl"
        # Live status (shared with backend /confluence/status)
        self.status_path = self.root / "logs" / f"confluence_status_{space}.json"

    def write_status(self, payload: Dict[str, Any]) -> None:
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            # Best-effort only; UI falls back to idle when unavailable
            pass


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_manifest(root: Path, suffixes: Tuple[str, ...]) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    if not root.exists():
        return manifest
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not path.suffix.lower() in suffixes:
            continue
        rel = path.relative_to(root).as_posix()
        try:
            manifest[rel] = _hash_file(path)
        except Exception:
            continue
    return manifest


def _load_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _write_json(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"warning: failed to write {path}: {exc}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_perf(log_path: Path, payload: Dict[str, Any]) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def evaluate(space: str, source_kind: str, source_dir: str) -> Plan:
    src = SourceSpec(kind=source_kind, path=str(Path(source_dir)))
    paths = SyncPaths(space)
    started = time.perf_counter()
    ts = _utc_now()
    suffixes = (".html",) if source_kind == "html" else (".md",)
    manifest = _make_manifest(Path(source_dir), suffixes)
    prev_state = _load_json(paths.state_path) or {}
    prev_manifest = prev_state.get("manifest") or {}

    added = sorted([k for k in manifest.keys() if k not in prev_manifest])
    changed = sorted([k for k, v in manifest.items() if prev_manifest.get(k) not in (None, v)])
    deleted = sorted([k for k in prev_manifest.keys() if k not in manifest])

    files = len(manifest)
    pages = files  # treat 1 file = 1 page in Stage 1 offline
    estimated_seconds = round(max(1.0, pages * 0.05), 1)

    plan = Plan(
        space=space,
        source=src,
        pages=pages,
        files=files,
        estimated_seconds=estimated_seconds,
        added=added,
        changed=changed,
        deleted=deleted,
    )
    _write_json(paths.plan_path, asdict(plan))
    _write_json(paths.delta_path, {"added": added, "changed": changed, "deleted": deleted})
    # Persist a snapshot of the manifest as the next base for delta (read-only until pull)
    _write_json(paths.state_path, {"space": space, "source": asdict(src), "manifest": manifest})
    elapsed = time.perf_counter() - started
    _append_perf(paths.perf_log, {
        "event": "evaluate_local",
        "space": space,
        "ts": ts,
        "duration_sec": round(elapsed, 3),
        "files": files,
        "pages": pages,
        "added": len(added),
        "changed": len(changed),
        "deleted": len(deleted),
        "source_kind": source_kind,
        "source_path": src.path,
    })
    print(
        f"Evaluate complete — space={space} kind={source_kind} files={files} pages={pages} "
        f"eta≈{estimated_seconds}s added={len(added)} changed={len(changed)} deleted={len(deleted)}"
    )
    return plan


# ------------------------------ Remote (Confluence) ---------------------------

def _slugify(title: str) -> str:
    base = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "-" for ch in (title or "").strip())
    base = "-".join([seg for seg in base.split("-") if seg])
    return base.lower() or "page"


class ConfluenceClient:
    def __init__(self, *, url: str, token: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, cloud: bool = False) -> None:
        if Confluence is None:
            raise SystemExit("The 'atlassian-python-api' package is required; add it to requirements and install.")
        # For Data Center PAT, pass token=PAT (bearer handled by library)
        # For basic auth, pass username/password.
        kwargs: Dict[str, Any] = {"url": url}
        if token:
            kwargs["token"] = token
            # Explicitly indicate non-cloud when using Data Center PAT
            kwargs["cloud"] = bool(cloud)
        elif username and password:
            kwargs["username"] = username
            kwargs["password"] = password
            kwargs["cloud"] = bool(cloud)
        else:
            raise SystemExit("Provide token (PAT) or username/password for Confluence client")
        self._client = Confluence(**kwargs)

    # Thin wrappers around library methods ---------------------------------
    def get_page(self, page_id: str, expand: Optional[str] = None) -> Dict[str, Any]:
        return self._client.get_page_by_id(page_id, expand=expand)

    def get_space_homepage_id(self, space_key: str) -> Optional[str]:
        try:
            sp = self._client.get_space(space_key, expand='homepage') or {}
            hid = ((sp.get('homepage') or {}).get('id'))
            if hid:
                return str(hid)
        except Exception:
            # Fallback to a conventional Home page lookup
            try:
                hid = self._client.get_page_id(space_key, 'Home')
                if hid:
                    return str(hid)
            except Exception:
                return None
        return None

    def list_children(self, page_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        # Prefer paginated child retrieval via type API; fallback to simple list
        out: List[Dict[str, Any]] = []
        try:
            start = 0
            while True:
                batch = self._client.get_page_child_by_type(page_id, type='page', start=start, limit=limit, expand='space') or []
                if not batch:
                    break
                out.extend(batch)
                if len(batch) < limit:
                    break
                start += len(batch)
            return out
        except Exception:
            try:
                batch = self._client.get_child_pages(page_id, expand='space') or []
                return batch
            except Exception:
                return []


def _load_auth(auth_file: Optional[str]) -> Dict[str, Any]:
    # Reads auth from JSON file; expected shape like auth.json in repo.
    path = Path(auth_file or ROOT / "auth.json")
    if not path.exists():
        raise SystemExit(f"Auth file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to parse auth file {path}: {exc}") from exc
    return data


def _resolve_dc_url_from_auth(auth: Dict[str, Any]) -> str:
    host = (((auth.get("servers") or {}).get("confluence") or {}).get("env") or {}).get("CONFLUENCE_HOST")
    if not host:
        raise SystemExit("CONFLUENCE_HOST missing in auth file under servers.confluence.env")
    host = str(host)
    if host.startswith("http://") or host.startswith("https://"):
        return host
    return f"https://{host}"


def _resolve_dc_token_from_auth(auth: Dict[str, Any]) -> str:
    token = (((auth.get("servers") or {}).get("confluence") or {}).get("env") or {}).get("CONFLUENCE_API_TOKEN")
    if not token:
        raise SystemExit("CONFLUENCE_API_TOKEN missing in auth file under servers.confluence.env")
    return str(token)


def evaluate_remote(
    space: str,
    root_page_id: str,
    *,
    auth_file: Optional[str] = None,
    workers: int = 10,
) -> Plan:
    paths = SyncPaths(space)
    started = time.perf_counter()
    started_wall = time.time()
    ts = _utc_now()
    auth = _load_auth(auth_file)
    base_url = _resolve_dc_url_from_auth(auth)
    token = _resolve_dc_token_from_auth(auth)
    client = ConfluenceClient(url=base_url, token=token, cloud=False)
    # Derive root page id dynamically when not provided or blank
    if not str(root_page_id or '').strip():
        derived = client.get_space_homepage_id(space)
        if not derived:
            raise SystemExit(f"Unable to resolve homepage for space {space}")
        root_page_id = derived
    manifest: Dict[str, Dict[str, Any]] = {}
    tree_edges: Dict[str, List[str]] = {}

    # Include root page itself (baseline metadata)
    root = client.get_page(root_page_id, expand="version,space")
    try:
        rid = str(root.get("id"))
    except Exception:
        rid = str(root_page_id)
    rver = (root.get("version") or {}).get("number")
    rspace_key = ((root.get("space") or {}).get("key") or "").strip()
    rtitle = root.get("title") or "Root"
    manifest[rid] = {"version": int(rver or 1), "title": rtitle, "space": rspace_key}

    # Phase 1: Traverse to collect all descendant page ids (serial listing, lightweight)
    all_ids: List[str] = [rid]
    id_to_title: Dict[str, str] = {rid: rtitle}
    queue: List[str] = [rid]
    seen: set[str] = {rid}
    visited = 0
    last_emit = time.time()
    # Emit initial traverse heartbeat
    paths.write_status({
        "space": space,
        "stage": "evaluate",
        "phase": "traverse",
        "progress": {"total": 1, "done": 0},
        "started_at": started_wall,
        "eta_sec": 0.0,
        "ts": last_emit,
    })
    while queue:
        current = queue.pop(0)
        visited += 1
        children = client.list_children(current) or []
        tree_edges.setdefault(str(current), [])
        for child in children:
            pid = str(child.get("id") or "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            # Keep only children in target space when available on listing
            spk = str(((child.get("space") or {}).get("key") or "")).strip().upper()
            if spk and rspace_key and spk != rspace_key.upper():
                # skip cross-space links
                continue
            all_ids.append(pid)
            title = str(child.get("title") or "")
            if title:
                id_to_title[pid] = title
            queue.append(pid)
            tree_edges.setdefault(str(current), []).append(pid)
        # Heartbeat every ~0.75s during traversal with approximate totals
        now_hb = time.time()
        if (now_hb - last_emit) >= 0.75:
            approx_total = max(visited + len(queue), visited)
            paths.write_status({
                "space": space,
                "stage": "evaluate",
                "phase": "traverse",
                "progress": {"total": approx_total, "done": visited},
                "started_at": started_wall,
                "eta_sec": 0.0,
                "ts": now_hb,
            })
            last_emit = now_hb

    total = len(all_ids)
    # Emit initial progress for Evaluate (metadata phase)
    paths.write_status({
        "space": space,
        "stage": "evaluate",
        "progress": {"total": total, "done": 0},
        "phase": "metadata",
        "started_at": started_wall,
        "ts": time.time(),
    })

    # Phase 2: Fetch versions for all pages concurrently (default 10 workers)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_meta(pid: str) -> Tuple[str, int, str, str]:
        try:
            data = ConfluenceClient(url=base_url, token=token, cloud=False).get_page(pid, expand="version,space") or {}
            ver = int(((data.get("version") or {}).get("number") or 1))
            title = str(data.get("title") or id_to_title.get(pid) or "")
            sp_key = str(((data.get("space") or {}).get("key") or "")).strip()
            return pid, ver, title, sp_key
        except Exception:
            # Fallback when metadata fails; keep title from listing
            return pid, 1, (id_to_title.get(pid) or ""), ""

    workers = max(1, min(int(workers or 10), 32))
    done = 0
    update_every = max(1, total // 100)  # up to ~100 updates
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_fetch_meta, pid) for pid in all_ids]
        for fut in as_completed(futures):
            pid, ver, title, sp_key = fut.result()
            manifest[pid] = {"version": int(ver or 1), "title": title, "space": sp_key}
            done += 1
            if (done % update_every == 0) or (done == total):
                elapsed = time.time() - started_wall
                eta = (total - done) * (elapsed / done) if done > 0 and total > done else 0.0
                paths.write_status({
                    "space": space,
                    "stage": "evaluate",
                    "progress": {"total": total, "done": done},
                    "phase": "metadata",
                    "started_at": started_wall,
                    "eta_sec": max(0.0, float(eta)),
                    "ts": time.time(),
                })

    prev = _load_json(paths.state_path) or {}
    prev_manifest = prev.get("manifest") or {}
    # Filter to the target space only (fall back to root space key if provided)
    target_space = (rspace_key or space or "").strip().upper()
    final_manifest: Dict[str, Dict[str, Any]] = {}
    for pid, meta in manifest.items():
        spk = str(meta.get("space") or "").strip().upper()
        if not spk:
            # conservatively keep when space key missing
            final_manifest[pid] = meta
        elif spk == target_space:
            final_manifest[pid] = meta
    # Compare by id+version against filtered set
    added = sorted([pid for pid in final_manifest.keys() if pid not in prev_manifest])
    changed = sorted([pid for pid, meta in final_manifest.items() if (prev_manifest.get(pid) or {}).get("version") not in (None, meta.get("version"))])
    deleted = sorted([pid for pid in prev_manifest.keys() if pid not in final_manifest])

    plan = Plan(
        space=space,
        source=SourceSpec(kind="remote", path=base_url),
        pages=len(final_manifest),
        files=len(final_manifest),
        estimated_seconds=round(max(1.0, len(manifest) * 0.08), 1),
        added=added,
        changed=changed,
        deleted=deleted,
    )
    _write_json(paths.plan_path, asdict(plan))
    _write_json(paths.delta_path, {"added": added, "changed": changed, "deleted": deleted})
    # Persist manifest without secrets
    _write_json(paths.state_path, {
        "space": space,
        "kind": "remote",
        "base_url": base_url,
        "root_page_id": root_page_id,
        "manifest": final_manifest,
    })
    # Persist tree adjacency for UI picker reuse
    try:
        allowed = set(final_manifest.keys()) | {str(rid)}
        tree_payload = {
            "root_page_id": str(rid),
            "nodes": {pid: {"title": (manifest.get(pid) or {}).get("title"), "children": [c for c in tree_edges.get(pid, []) if c in allowed]} for pid in ([str(rid)] + list(final_manifest.keys()))},
        }
        (paths.state_dir / "tree.json").write_text(json.dumps(tree_payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    elapsed = time.perf_counter() - started
    _append_perf(paths.perf_log, {
        "event": "evaluate_remote",
        "space": space,
        "ts": ts,
        "duration_sec": round(elapsed, 3),
        "pages": len(final_manifest),
        "added": len(added),
        "changed": len(changed),
        "deleted": len(deleted),
        "root_page_id": str(root_page_id),
        "base_url": base_url,
        "workers": workers,
    })
    print(
        f"Evaluate complete (remote) — space={space} pages={len(manifest)} eta≈{plan.estimated_seconds}s "
        f"added={len(added)} changed={len(changed)} deleted={len(deleted)}"
    )
    # Final status snapshot (backend will flip to idle on thread completion)
    paths.write_status({
        "space": space,
        "stage": "evaluate",
        "progress": {"total": total, "done": total},
        "phase": "metadata",
        "started_at": started_wall,
        "elapsed_sec": round(elapsed, 3),
        "eta_sec": 0.0,
        "ts": time.time(),
    })
    return plan


def _copy_tree(src: Path, dst: Path, suffixes: Tuple[str, ...]) -> int:
    copied = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        copied += 1
    return copied


def _fetch_page_html(base_url: str, token: str, pid: str, title: str, out_dir: Path, *, retries: int = 5, backoff: float = 0.5, stop_labels: Optional[List[str]] = None) -> Dict[str, Any]:
    last_error: Optional[str] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            client = ConfluenceClient(url=base_url, token=token, cloud=False)
            data = client.get_page(pid, expand="body.storage,version,metadata.labels") or {}
            # Optional label-based exclusion
            if stop_labels:
                try:
                    block = ((data.get("metadata") or {}).get("labels") or {})
                    results = block.get("results") or []
                    page_labels = {str((r.get("name") or "")).strip().lower() for r in results}
                except Exception:
                    page_labels = set()
                norm = {str(x).strip().lower() for x in stop_labels if str(x).strip()}
                if page_labels and norm.intersection(page_labels):
                    return {"ok": True, "pid": pid, "title": title, "skipped": True, "cause": "label_excluded"}
            html_value = (((data.get("body") or {}).get("storage") or {}).get("value") or "").strip()
            slug = _slugify(title or data.get("title") or pid)
            filename = f"{slug}_{pid}.html"
            out = out_dir / filename
            out.parent.mkdir(parents=True, exist_ok=True)
            if html_value:
                out.write_text(html_value, encoding="utf-8")
                return {"ok": True, "pid": pid, "title": title, "attempts": attempt}
            # No storage content — write a minimal placeholder and treat as success
            placeholder = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>{title or data.get('title') or pid}</title></head>
<body>
<h1>{title or data.get('title') or pid}</h1>
<p>(No storage content returned; placeholder stub preserved for hierarchy.)</p>
</body></html>"""
            out.write_text(placeholder, encoding="utf-8")
            return {"ok": True, "pid": pid, "title": title, "attempts": attempt, "cause": "empty_storage_stub"}
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc) or last_error or "unknown error"
            # simple backoff
            try:
                import time as _time
                _time.sleep(backoff * attempt)
            except Exception:
                pass
    return {"ok": False, "pid": pid, "title": title, "attempts": retries, "error": last_error}


def pull(space: str, *, auth_file: Optional[str] = None, workers: int = 10, retries: int = 5) -> None:
    paths = SyncPaths(space)
    state = _load_json(paths.state_path) or {}
    kind = (state.get("kind") or (state.get("source") or {}).get("kind") or "").lower()

    if kind == "remote" or auth_file:
        # Remote pull: fetch added/changed pages using atlassian Confluence client
        started = time.perf_counter()
        started_wall = time.time()
        ts = _utc_now()
        auth = _load_auth(auth_file)
        base_url = state.get("base_url") or _resolve_dc_url_from_auth(auth)
        token = _resolve_dc_token_from_auth(auth)
        client = ConfluenceClient(url=str(base_url), token=str(token), cloud=False)
        # Ensure plan exists
        plan = _load_json(paths.plan_path) or {}
        if not plan or plan.get("space") != space:
            rpid = state.get("root_page_id") or ""
            if not rpid:
                raise SystemExit("root_page_id missing in state; run evaluate first")
            evaluate_remote(space, rpid, auth_file=auth_file)
            plan = _load_json(paths.plan_path) or {}
        # Respect saved selection when present
        sel_path = paths.state_dir / "selection.json"
        selection_ids: set[str] = set()
        stop_labels: List[str] = []
        stop_words: List[str] = []
        try:
            sel = _load_json(sel_path) or {}
            selection_ids = set(sel.get("include_ids") or [])
            stop_labels = [str(x).strip() for x in (sel.get("stop_labels") or []) if str(x).strip()]
            stop_words = [str(x).strip().lower() for x in (sel.get("stop_words") or []) if str(x).strip()]
        except Exception:
            selection_ids = set()

        delta_ids = set(plan.get("added", [])) | set(plan.get("changed", []))
        # Full pull of selected pages when a selection is present; otherwise use delta; fallback to manifest
        if selection_ids:
            ids_to_fetch = selection_ids
            mode = "selection"
        elif delta_ids:
            ids_to_fetch = delta_ids
            mode = "delta"
        else:
            ids_to_fetch = set((state.get("manifest") or {}).keys())
            mode = "manifest"
        # Optional stop-words filter (title only)
        if stop_words:
            manifest = (state.get("manifest") or {})
            norm_sw = [w for w in stop_words if w]
            filtered: set[str] = set()
            for pid in ids_to_fetch:
                title = str(((manifest.get(pid) or {}).get("title") or "")).strip().lower()
                if title and any(sw in title for sw in norm_sw):
                    continue
                filtered.add(pid)
            ids_to_fetch = filtered
        print(f"Pull selection: mode={mode} selected={len(selection_ids)} delta={len(delta_ids)} => to_fetch={len(ids_to_fetch)}", flush=True)
        if not ids_to_fetch:
            print("Pull skipped — no added/changed pages.")
            return
        paths.staging_html.mkdir(parents=True, exist_ok=True)
        manifest = (state.get("manifest") or {})
        # Multi-threaded fetch
        from concurrent.futures import ThreadPoolExecutor, as_completed
        page_meta = [(pid, (manifest.get(pid) or {}).get("title") or pid) for pid in sorted(ids_to_fetch)]
        total = len(page_meta)
        successes = 0
        failures = 0
        excluded_by_label = 0
        failed_items: List[Dict[str, Any]] = []
        workers = max(1, min(int(workers or 10), 16))
        print(f"Starting pull with {workers} worker(s) — {total} page(s)…", flush=True)
        # Initial status snapshot
        paths.write_status({
            "space": space,
            "stage": "pull",
            "progress": {"total": total, "done": 0, "failed": 0},
            "phase": "fetch",
            "started_at": started_wall,
            "eta_sec": 0.0,
            "ts": time.time(),
        })
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _fetch_page_html,
                    str(base_url),
                    str(token),
                    pid,
                    title,
                    paths.staging_html,
                    retries=retries,
                    stop_labels=stop_labels,
                )
                for pid, title in page_meta
            ]
            for fut in as_completed(futures):
                res = fut.result()
                ok = bool(res.get("ok"))
                if ok:
                    if res.get("skipped") and res.get("cause") == "label_excluded":
                        excluded_by_label += 1
                    else:
                        successes += 1
                else:
                    failures += 1
                    item = {
                        "pid": res.get("pid"),
                        "title": res.get("title"),
                        "error": res.get("error") or "unknown",
                        "attempts": res.get("attempts"),
                    }
                    failed_items.append(item)
                    # Highlight failure in logs (single line, flush)
                    print(
                        f"FAILED pid={item['pid']} attempts={item['attempts']} reason={item['error']}",
                        flush=True,
                    )
                done = successes + failures
                # Periodic CLI print and status update
                if (done % 10 == 0) or (done == total):
                    print(f"progress: {done}/{total} fetched={successes} failed={failures}", flush=True)
                    elapsed = time.time() - started_wall
                    eta = (total - done) * (elapsed / done) if done > 0 and total > done else 0.0
                    paths.write_status({
                        "space": space,
                        "stage": "pull",
                        "progress": {"total": total, "done": done, "failed": failures},
                        "excluded_by_label": excluded_by_label,
                        "phase": "fetch",
                        "started_at": started_wall,
                        "eta_sec": max(0.0, float(eta)),
                        "ts": time.time(),
                    })
        # Persist failures summary
        if failed_items:
            fail_path = ROOT / "logs" / "confluence_pull_failed.json"
            try:
                fail_path.parent.mkdir(parents=True, exist_ok=True)
                fail_path.write_text(json.dumps(failed_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(f"Failed items saved to {fail_path}", flush=True)
            except Exception:
                pass
        print(f"Pull complete (remote) — fetched {successes} / {total} page(s) to {paths.staging_html} (failed={failures})", flush=True)
        # Final status snapshot
        paths.write_status({
            "space": space,
            "stage": "pull",
            "progress": {"total": total, "done": total, "failed": failures},
            "excluded_by_label": excluded_by_label,
            "phase": "fetch",
            "started_at": started_wall,
            "elapsed_sec": round(time.time() - started_wall, 3),
            "eta_sec": 0.0,
            "ts": time.time(),
        })
        # Perf
        elapsed = time.perf_counter() - started
        _append_perf(paths.perf_log, {
            "event": "pull_remote",
            "space": space,
            "ts": ts,
            "duration_sec": round(elapsed, 3),
            "total": total,
            "fetched": successes,
            "failed": failures,
            "workers": workers,
            "retries": retries,
        })
        return

    # Local copy (reference)
    src = state.get("source") or {}
    source_kind = (src.get("kind") or "").lower()
    source_dir = src.get("path")
    if not source_kind or not source_dir:
        raise SystemExit("No evaluate state found; run 'evaluate' first or provide valid source.")
    src_path = Path(source_dir)
    if not src_path.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")
    if source_kind == "html":
        dst = paths.staging_html
        suffixes = (".html",)
    elif source_kind == "md":
        dst = paths.staging_md
        suffixes = (".md",)
    else:
        raise SystemExit(f"Unsupported source kind: {source_kind}")
    if dst.exists():
        print(f"Staging directory exists, updating: {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    count = _copy_tree(src_path, dst, suffixes)
    print(f"Pull complete — copied {count} file(s) to {dst}")


def build(space: str) -> None:
    paths = SyncPaths(space)
    # Convert HTML→MD when needed
    build_started_wall = time.time()
    if paths.staging_html.exists():
        if convert_html_to_md is None:
            raise SystemExit("convert_html_to_md.py not available")
        print(f"Converting HTML → MD: {paths.staging_html} -> {paths.staging_md}")
        # Status: build phase chunk (conversion)
        try:
            paths.write_status({
                "space": space,
                "stage": "build",
                "phase": "chunk",
                "ts": time.time(),
                "started_at": build_started_wall,
            })
        except Exception:
            pass
        convert_html_to_md(str(paths.staging_html), str(paths.staging_md))

    if not paths.staging_md.exists():
        raise SystemExit(f"Staging Markdown directory not found: {paths.staging_md}")

    # Prune non-selected MD files to ensure staging strictly matches selection
    try:
        sel = _load_json(paths.state_dir / "selection.json") or {}
        include_ids = set(sel.get("include_ids") or [])
        if include_ids and paths.staging_md.exists():
            import re as _re
            pat = _re.compile(r"_([0-9]+)\.md$", _re.IGNORECASE)
            removed = 0
            for md in paths.staging_md.rglob("*.md"):
                m = pat.search(md.name)
                pid = m.group(1) if m else None
                if pid and pid not in include_ids:
                    try:
                        md.unlink()
                        removed += 1
                    except Exception:
                        pass
            if removed:
                print(f"Pruned {removed} non-selected markdown files from staging.")
    except Exception:
        pass

    # Chunk Markdown into staging chunks file
    if chunk_confluence_markdown is None:
        raise SystemExit("chunk_confluence.py not available")
    chunk_ts = _utc_now()
    chunk_started = time.perf_counter()
    print(f"Chunking Markdown: {paths.staging_md} -> {paths.staging_chunks}")
    chunk_confluence_markdown(str(paths.staging_md), str(paths.staging_chunks))
    chunk_elapsed = time.perf_counter() - chunk_started
    print(f"Chunking complete — wrote {paths.staging_chunks}")
    _append_perf(paths.perf_log, {
        "event": "chunk_markdown",
        "space": space,
        "ts": chunk_ts,
        "duration_sec": round(chunk_elapsed, 3),
        "md_dir": str(paths.staging_md),
        "chunks_path": str(paths.staging_chunks),
    })

    # Build staging FAISS index (confluence_staging) without touching active indexes
    try:
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"FAISS/sentence-transformers unavailable: {exc}")

    def _iter_confluence_records(chunks: List[Dict[str, Any]]):
        for chunk in chunks:
            content = (chunk.get("content") or "").strip()
            if not content:
                continue
            heading_path = chunk.get("heading_path", [])
            path_text = " > ".join(heading_path)
            page_title = chunk.get("page_title", "")
            uid = chunk.get("chunk_id") or f"{chunk.get('page_id','unknown')}::{(chunk.get('source') or {}).get('start_line',0)}"
            metadata = {
                "type": "confluence_chunk",
                "page_id": chunk.get("page_id"),
                "page_title": page_title,
                "heading_path": heading_path,
                "source": chunk.get("source", {}),
            }
            text = f"{page_title}\n{path_text}\n{content}" if path_text else f"{page_title}\n{content}"
            yield {"uid": uid, "text": text, "metadata": metadata}

    build_ts = _utc_now()
    build_started = time.perf_counter()
    chunks = json.loads(Path(paths.staging_chunks).read_text(encoding="utf-8"))
    records = list(_iter_confluence_records(chunks))
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    texts = [r["text"] for r in records]
    # Status: embedding batches
    batch_size = 64
    total_batches = (len(texts) + batch_size - 1) // batch_size
    done_batches = 0
    try:
        paths.write_status({
            "space": space,
            "stage": "build",
            "phase": "embed_faiss",
            "progress": {"total": total_batches, "done": 0},
            "started_at": build_started_wall,
            "ts": time.time(),
        })
    except Exception:
        pass
    # Encode in batches and build index incrementally
    dim = None
    index = None
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if not batch:
            continue
        emb = model.encode(batch, batch_size=len(batch), convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        if dim is None:
            dim = emb.shape[1] if emb.size else 768
            index = faiss.IndexFlatIP(dim)
        if emb.size:
            index.add(emb)
        done_batches += 1
        if done_batches % 2 == 0 or done_batches == total_batches:
            try:
                paths.write_status({
                    "space": space,
                    "stage": "build",
                    "phase": "embed_faiss",
                    "progress": {"total": total_batches, "done": done_batches},
                    "started_at": build_started_wall,
                    "ts": time.time(),
                })
            except Exception:
                pass
    if index is None:
        index = faiss.IndexFlatIP(768)
    vs_dir = ROOT / "vector_store"
    vs_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = vs_dir / "confluence_staging.faiss"
    meta_path = vs_dir / "confluence_staging_metadata.json"
    faiss.write_index(index, str(faiss_path))
    meta = {"model": "BAAI/bge-base-en-v1.5", "records": records}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
    build_elapsed = time.perf_counter() - build_started
    print(f"Built staging FAISS index at {faiss_path}")
    _append_perf(paths.perf_log, {
        "event": "build_staging_faiss",
        "space": space,
        "ts": build_ts,
        "duration_sec": round(build_elapsed, 3),
        "chunks": len(records),
        "faiss_path": str(faiss_path),
        "metadata_path": str(meta_path),
        "model": "BAAI/bge-base-en-v1.5",
    })


def _symlink_force(target: Path, link: Path) -> None:
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
    except Exception:
        pass
    try:
        link.symlink_to(target)
    except Exception:
        # Fallback: create a marker file with the target path when symlink unsupported
        link.write_text(str(target), encoding="utf-8")


def activate(space: str) -> None:
    paths = SyncPaths(space)
    if not paths.staging_chunks.exists():
        raise SystemExit(f"Staging chunks not found; run 'build' first: {paths.staging_chunks}")

    # Activate chunks for index build
    print(f"Activating chunks: {paths.staging_chunks} -> {paths.active_chunks}")
    shutil.copy2(paths.staging_chunks, paths.active_chunks)

    # Point active MD to the staging MD
    if paths.staging_md.exists():
        print(f"Updating active MD pointer: {paths.active_md_link} -> {paths.staging_md}")
        _symlink_force(paths.staging_md, paths.active_md_link)

    # Rebuild vector store (confluence index)
    print("Rebuilding vector store (confluence)…")
    # Status heartbeat while rebuilding
    try:
        activate_started_wall = time.time()
        paths.write_status({
            "space": space,
            "stage": "activate",
            "phase": "rebuild_full_index",
            "started_at": activate_started_wall,
            "progress": {"total": 100, "done": 0},
            "ts": time.time(),
        })
    except Exception:
        pass
    import re
    percent_re = re.compile(r"(\d+)%\|")
    cmd = [sys.executable, str(ROOT / "build_vector_store.py")]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # noqa: S603,S607
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            print(line)
            m = percent_re.search(line)
            if m:
                try:
                    pct = int(m.group(1))
                    paths.write_status({
                        "space": space,
                        "stage": "activate",
                        "phase": "rebuild_full_index",
                        "progress": {"total": 100, "done": pct},
                        "started_at": activate_started_wall,
                        "ts": time.time(),
                    })
                except Exception:
                    pass
    finally:
        ret = proc.wait()
        if ret != 0:
            raise SystemExit(f"Vector store build failed with exit code {ret}")
    print("Activation complete — confluence index refreshed.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Confluence Stage 1 offline sync")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Evaluate Confluence (remote, multi-threaded) or a local export and produce a plan/delta")
    p_eval.add_argument("--space", required=True, help="Logical space identifier (e.g., SWT1AQ)")
    g = p_eval.add_mutually_exclusive_group(required=True)
    g.add_argument("--auth-file", dest="auth_file", help="Path to auth JSON (Data Center PAT)")
    g.add_argument("--source-html", dest="source_html", help="Local HTML source directory (reference mode)")
    g.add_argument("--source-md", dest="source_md", help="Local Markdown source directory (reference mode)")
    p_eval.add_argument("--root-page-id", help="Root page id to traverse (remote)")
    p_eval.add_argument("--workers", type=int, default=10, help="Concurrent metadata fetch workers for remote evaluate (default 10)")

    p_pull = sub.add_parser("pull", help="Fetch from Confluence into staging (remote) or copy local export (reference)")
    p_pull.add_argument("--space", required=True, help="Logical space identifier (e.g., SWT1AQ)")
    p_pull.add_argument("--auth-file", help="Path to auth JSON (Data Center PAT)")
    p_pull.add_argument("--workers", type=int, default=10, help="Concurrent fetch workers (1-16)")

    p_build = sub.add_parser("build", help="Convert (if HTML) and chunk Markdown into staging chunks JSON")
    p_build.add_argument("--space", required=True, help="Logical space identifier (e.g., SWT1AQ)")

    p_activate = sub.add_parser("activate", help="Activate staging dataset and rebuild confluence index")
    p_activate.add_argument("--space", required=True, help="Logical space identifier (e.g., SWT1AQ)")

    args = parser.parse_args(argv)

    if args.command == "evaluate":
        if args.auth_file:
            if not args.root_page_id:
                parser.error("--root-page-id is required for remote evaluate")
            evaluate_remote(
                args.space,
                args.root_page_id,
                auth_file=args.auth_file,
                workers=int(args.workers or 10),
            )
        elif args.source_html:
            evaluate(args.space, "html", args.source_html)
        elif args.source_md:
            evaluate(args.space, "md", args.source_md)
        else:
            # fallback: default auth file
            if args.root_page_id:
                evaluate_remote(args.space, args.root_page_id, auth_file=str(ROOT / "auth.json"), workers=int(args.workers or 10))
            else:
                parser.error("Provide --auth-file and --root-page-id (remote) or --source-* (local)")
    elif args.command == "pull":
        pull(args.space, auth_file=args.auth_file, workers=int(args.workers or 4))
    elif args.command == "build":
        build(args.space)
    elif args.command == "activate":
        activate(args.space)
    else:
        parser.error("unknown command")


if __name__ == "__main__":
    main()
