import uvicorn
import sys
import os
import json
import requests
import time
from collections import deque
from fastapi import APIRouter, Body, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect, Query
from server.searching_server import SearchingServer
from server.schemas import StrRequestModel, ContentItemModel, DialogueParams, Message
from server.dialogue import Dialogue_gpt2, Conversation
from typing import Any, Dict, List, Literal, Optional, Tuple, Set
import logging
from pydantic import BaseModel
from pathlib import  Path

import threading
import uuid as _uuid
import requests


APP_ROOT = Path(__file__).resolve().parent.parent
CORPUS_ROOT = APP_ROOT / "corpus"
_CONF_LOG = logging.getLogger("backend.confluence")
_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _ROOT / "scripts"

_CONF_LOG.info("_ROOT: " + str(_ROOT))

import sys as _sys
_sys.path.insert(0, str(_SCRIPTS_DIR))
from confluence_sync import (
    evaluate_remote as cf_evaluate_remote,
    pull as cf_pull,
    build as cf_build,
    activate as cf_activate,
    ConfluenceClient as CfClient,
)  # type: ignore

app = FastAPI()

searching_server_global = SearchingServer()

conversations = {
    "developer": Conversation(system_prompt="You are developer Assistant."),
    "manager": Conversation(system_prompt="You are project-manager Assistant."),
    "auditor": Conversation(system_prompt="You are auditor Assistant."),
}

dialogue_dev = Dialogue_gpt2()

dialogue_dev.handle_user_message(conversations["developer"])
# dialogue_dev.handle_user_message(conversations["manager"])
# dialogue_dev.handle_user_message(conversations["auditor"])

######################################################
def searching_server():
    global searching_server_global
    return searching_server_global


@app.post("/ai-search", response_model=List[ContentItemModel])
async def ai_search(input_request: StrRequestModel):
    return searching_server().search(input_request.str_request)


@app.post("/page-to-index")
async def page_to_index(input_request: StrRequestModel):
    searching_server().page_to_index(input_request.str_request)
    return { "status": "200" }


@app.post("/text-to-index")
async def text_to_index(input_request: StrRequestModel):
    searching_server().text_to_index(input_request.str_request)
    return { "status": "200" }


@app.post("/new-message")
async def new_message(dialogue: DialogueParams):
    if dialogue.dialogue_type in {"developer", "manager", "auditor",}:
        print("new message:", dialogue.message_str)
        dialogue_dev.handle_user_message(
            conversations[dialogue.dialogue_type],
            dialogue.message_str
            )
    return { "status": "200" }


@app.post("/get-last-answer", response_model=Optional[Message])
async def get_last_answer(dialogue: DialogueParams):
    if dialogue.dialogue_type in {"developer", "manager", "auditor",}:
        return dialogue_dev.get_last_answer(
            conversations[dialogue.dialogue_type]
            )
    else:
        return None


@app.post("/get-dialogue", response_model=List[Message])
async def get_dialogue(dialogue: DialogueParams):
    if dialogue.dialogue_type in {"developer", "manager", "auditor",}:
        return dialogue_dev.get_messages(
            conversations[dialogue.dialogue_type]
            )
    else:
        return []


@app.post("/clear-dialogue")
async def clear_dialogue(dialogue: DialogueParams):
    if dialogue.dialogue_type in {"developer", "manager", "auditor",}:
        #print("::clear_dialogue:", dialogue.dialogue_type)
        dialogue_dev.clear(
            conversations[dialogue.dialogue_type]
            )
    return { "status": "200" }



# --------------------------------------------------------
# Confluence Stage 2 — Backend endpoints (PAT, tree, pipeline)
# --------------------------------------------------------

class ConfluencePatRequest(BaseModel):
    confluence_host: str
    token: str
    root_page_id: str | None = None


class ConfluencePatSaveRequest(BaseModel):
    confluence_host: str
    token: str | None = None


class ConfluenceEvaluateRequest(BaseModel):
    space: str
    root_page_id: str
    workers: int | None = 10


class ConfluencePullRequest(BaseModel):
    space: str
    workers: int | None = 10


class ConfluenceBuildRequest(BaseModel):
    space: str


class ConfluenceActivateRequest(BaseModel):
    space: str


class ConfluenceSelectionRequest(BaseModel):
    space: str
    include_ids: list[str]
    exclude_ids: list[str] | None = None
    # Optional filters persisted alongside selection
    stop_labels: list[str] | None = None
    stop_words: list[str] | None = None


_TASKS: dict[str, dict[str, Any]] = {}


def _secrets_path() -> Path:
    p = APP_ROOT / "secrets" / "confluence.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_space_auth(space: str, host: str, token: str) -> Path:
    payload = {
        "servers": {"confluence": {"env": {"CONFLUENCE_HOST": host, "CONFLUENCE_API_TOKEN": token}}}
    }
    path = (CORPUS_ROOT / space / ".sync")
    path.mkdir(parents=True, exist_ok=True)
    auth_path = path / "auth.json"
    auth_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return auth_path


def _load_saved_pat() -> dict[str, str] | None:
    try:
        data = json.loads(_secrets_path().read_text(encoding="utf-8"))
        host = str(data.get("confluence_host") or "").strip()
        token = str(data.get("token") or "").strip()
        if host and token:
            return {"confluence_host": host, "token": token}
    except Exception:
        return None
    return None


@app.post("/confluence/pat/test", tags=["confluence"])
def confluence_pat_test(req: ConfluencePatRequest) -> Dict[str, Any]:
    host = req.confluence_host.strip()
    token = req.token.strip()
    try:
        # Lightweight DC test via REST: list spaces (limit 1)
        url = host.rstrip("/") + "/rest/api/space?limit=1"
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10)
        ok = 200 <= resp.status_code < 300
        if not ok:
            return {"ok": False, "error": f"HTTP {resp.status_code}"}
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/confluence/pat", tags=["confluence"])
def confluence_pat_get() -> Dict[str, Any]:
    saved = _load_saved_pat() or {}
    host = str(saved.get("confluence_host") or "").strip()
    has_token = bool(saved.get("token") or "")
    return {"ok": True, "confluence_host": host, "has_token": has_token}


@app.post("/confluence/pat/save", tags=["confluence"])
def confluence_pat_save(req: ConfluencePatSaveRequest) -> Dict[str, Any]:
    cur = _load_saved_pat() or {}
    host = req.confluence_host.strip()
    token = (req.token or "").strip()
    if not host:
        raise HTTPException(status_code=400, detail="confluence_host required")
    if not token:
        # keep existing token if not provided
        token = str(cur.get("token") or "").strip()
    data = {"confluence_host": host, "token": token}
    _secrets_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"ok": True}


@app.get("/confluence/tree", tags=["confluence"])
def confluence_tree(root_page_id: str | None = None, space: str | None = None, limit: int = 200) -> Dict[str, Any]:
    saved = _load_saved_pat()
    if not saved:
        raise HTTPException(status_code=400, detail="PAT not configured")
    host = saved["confluence_host"]
    token = saved["token"]
    try:
        _CONF_LOG.info("tree request space=%s root_page_id=%s limit=%s", space, root_page_id, limit)
        client = CfClient(url=host, token=token, cloud=False)
        rid = (root_page_id or "").strip()
        if not rid:
            # Resolve space home page id
            if not space:
                raise HTTPException(status_code=400, detail="Provide 'space' or 'root_page_id'")
            # Try direct space homepage
            try:
                sp = client._client.get_space(space, expand='homepage')  # type: ignore[attr-defined]
                rid = str(((sp or {}).get('homepage') or {}).get('id') or '').strip()
            except Exception:
                rid = ''
            # Fallback to 'Home' title lookup
            if not rid:
                try:
                    rid = str(client._client.get_page_id(space, 'Home'))  # type: ignore[attr-defined]
                except Exception:
                    rid = ''
            if not rid:
                raise HTTPException(status_code=404, detail=f"Unable to resolve homepage for space {space}")
        children = client.list_children(rid, limit=limit) or []
        nodes: list[dict[str, Any]] = []
        for c in children:
            pid = str(c.get("id") or "")
            title = str(c.get("title") or "")
            nodes.append({"id": pid, "title": title})
        _CONF_LOG.info("tree resolved rid=%s children=%d", rid, len(nodes))
        return {"ok": True, "root_page_id": rid, "children": nodes}
    except Exception as exc:  # noqa: BLE001
        _CONF_LOG.exception("tree error space=%s root=%s err=%s", space, root_page_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/confluence/tree/full", tags=["confluence"])
def confluence_tree_full(space: str) -> Dict[str, Any]:
    sync_dir = CORPUS_ROOT / space / ".sync"
    tree_path = sync_dir / "tree.json"
    try:
        text = tree_path.read_text(encoding="utf-8")
        data = json.loads(text)
        nodes = data.get("nodes") or {}
        _CONF_LOG.info("tree_full space=%s root=%s nodes=%d size=%dB", space, data.get("root_page_id"), len(nodes), len(text.encode('utf-8')))
    except Exception:
        _CONF_LOG.warning("tree_full missing for space=%s", space)
        raise HTTPException(status_code=404, detail="Full tree not available; run Evaluate first")
    return {"ok": True, **data}


@app.get("/confluence/history", tags=["confluence"])
def confluence_history(space: str, cmd: str = "pull", lines: int = 200) -> Dict[str, Any]:
    """Return the tail of the detached runner log for the given command.
    cmd: one of evaluate|pull|build|activate.
    """
    safe_cmd = cmd if cmd in {"evaluate", "pull", "build", "activate"} else "pull"
    safe_space = re.sub(r"[^A-Za-z0-9._-]", "_", space)
    log_path = APP_ROOT / "logs" / f"confluence_{safe_cmd}_{safe_space}.log"
    try:
        content = log_path.read_text(encoding="utf-8").splitlines()
        tail = content[-max(1, int(lines)) :]
    except Exception:
        tail = []
    return {"ok": True, "cmd": safe_cmd, "space": space, "lines": tail}


@app.post("/confluence/selection", tags=["confluence"])
def confluence_selection(req: ConfluenceSelectionRequest) -> Dict[str, Any]:
    path = CORPUS_ROOT / req.space / ".sync"
    path.mkdir(parents=True, exist_ok=True)
    sel_path = path / "selection.json"
    sel = {"include_ids": req.include_ids, "exclude_ids": req.exclude_ids or []}
    if isinstance(req.stop_labels, list):
        sel["stop_labels"] = [str(x).strip() for x in req.stop_labels if str(x).strip()]
    if isinstance(req.stop_words, list):
        sel["stop_words"] = [str(x).strip() for x in req.stop_words if str(x).strip()]
    sel_path.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    return {"ok": True}


@app.get("/confluence/selection", tags=["confluence"])
def confluence_selection_get(space: str) -> Dict[str, Any]:
    sel_path = CORPUS_ROOT / space / ".sync" / "selection.json"
    try:
        data = json.loads(sel_path.read_text(encoding="utf-8"))
        include_ids = list((data.get("include_ids") or []))
        exclude_ids = list((data.get("exclude_ids") or []))
        resp: Dict[str, Any] = {"ok": True, "include_ids": include_ids, "exclude_ids": exclude_ids}
        # Optional filters when present
        if isinstance(data.get("stop_labels"), list):
            resp["stop_labels"] = [str(x) for x in data.get("stop_labels")]
        if isinstance(data.get("stop_words"), list):
            resp["stop_words"] = [str(x) for x in data.get("stop_words")]
        return resp
    except Exception:
        return {"ok": True, "include_ids": [], "exclude_ids": []}


def _update_status(space: str, stage: str, extra: Dict[str, Any] | None = None) -> None:
    st = {"space": space, "stage": stage, "ts": time.time()}
    if extra:
        st.update(extra)
    out = APP_ROOT / "logs" / f"confluence_status_{space}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(st), encoding="utf-8")


def _run_task(space: str, fn, *args, **kwargs) -> None:  # noqa: ANN001
    try:
        _TASKS.setdefault(space, {})
        _update_status(space, kwargs.pop("__stage", "running"))
        fn(*args, **kwargs)
        _update_status(space, "idle")
    except SystemExit as exc:
        _update_status(space, "error", {"error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        _update_status(space, "error", {"error": str(exc)})
    finally:
        try:
            _TASKS[space].pop("thread", None)
        except Exception:
            pass


@app.post("/confluence/evaluate", tags=["confluence"])
def confluence_evaluate(req: ConfluenceEvaluateRequest) -> Dict[str, Any]:
    saved = _load_saved_pat()
    if not saved:
        raise HTTPException(status_code=400, detail="PAT not configured")
    auth_path = _write_space_auth(req.space, saved["confluence_host"], saved["token"])
    # Derive the space homepage id when not provided
    rid = (req.root_page_id or "").strip()
    if not rid:
        try:
            client = CfClient(url=saved["confluence_host"], token=saved["token"], cloud=False)
            sp = client._client.get_space(req.space, expand='homepage')  # type: ignore[attr-defined]
            rid = str(((sp or {}).get('homepage') or {}).get('id') or '').strip()
        except Exception:
            rid = ''
        if not rid:
            try:
                rid = str(client._client.get_page_id(req.space, 'Home'))  # type: ignore[attr-defined]
            except Exception:
                rid = ''
        if not rid:
            raise HTTPException(status_code=404, detail=f"Unable to resolve homepage for space {req.space}")
    run_id = str(_uuid.uuid4())
    t = threading.Thread(
        target=_run_task,
        args=(req.space, cf_evaluate_remote, req.space, rid),
        kwargs={"auth_file": str(auth_path), "workers": int(req.workers or 10), "__stage": "evaluate"},
        daemon=True,
    )
    _TASKS.setdefault(req.space, {})["thread"] = t
    t.start()
    return {"ok": True, "run_id": run_id}


@app.post("/confluence/pull", tags=["confluence"])
def confluence_pull(req: ConfluencePullRequest) -> Dict[str, Any]:
    saved = _load_saved_pat()
    if not saved:
        raise HTTPException(status_code=400, detail="PAT not configured")
    auth_path = _write_space_auth(req.space, saved["confluence_host"], saved["token"])
    run_id = str(_uuid.uuid4())
    t = threading.Thread(
        target=_run_task,
        args=(req.space, cf_pull, req.space),
        kwargs={"auth_file": str(auth_path), "workers": int(req.workers or 10), "__stage": "pull"},
        daemon=True,
    )
    _TASKS.setdefault(req.space, {})["thread"] = t
    t.start()
    return {"ok": True, "run_id": run_id}


@app.post("/confluence/build", tags=["confluence"])
def confluence_build(req: ConfluenceBuildRequest) -> Dict[str, Any]:
    run_id = str(_uuid.uuid4())
    t = threading.Thread(
        target=_run_task,
        args=(req.space, cf_build, req.space),
        kwargs={"__stage": "build"},
        daemon=True,
    )
    _TASKS.setdefault(req.space, {})["thread"] = t
    t.start()
    return {"ok": True, "run_id": run_id}


@app.post("/confluence/activate", tags=["confluence"])
def confluence_activate(req: ConfluenceActivateRequest) -> Dict[str, Any]:
    run_id = str(_uuid.uuid4())
    t = threading.Thread(
        target=_run_task,
        args=(req.space, cf_activate, req.space),
        kwargs={"__stage": "activate"},
        daemon=True,
    )
    _TASKS.setdefault(req.space, {})["thread"] = t
    t.start()
    return {"ok": True, "run_id": run_id}


@app.get("/confluence/status", tags=["confluence"])
def confluence_status(space: str) -> Dict[str, Any]:
    # Compose status from state/plan + last status file
    sync_dir = CORPUS_ROOT / space / ".sync"
    plan = {}
    delta = {}
    try:
        plan = json.loads((sync_dir / "plan.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        delta = json.loads((sync_dir / "delta.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    st_path = APP_ROOT / "logs" / f"confluence_status_{space}.json"
    raw_status: Dict[str, Any] = {}
    stage = "idle"
    try:
        raw_status = json.loads(st_path.read_text(encoding="utf-8")) or {}
        stage = (raw_status.get("stage") or "idle")
    except Exception:
        raw_status = {}

    # If no background task is currently tracked, treat stage as idle to avoid stale lockouts
    # (e.g., server restarts during an evaluate/pull/build/activate run leaving a non-idle status file).
    try:
        t = (_TASKS.get(space) or {}).get("thread")
        # Consider recent external status updates as in-progress even if thread not tracked
        recent = False
        try:
            ts_val = float(raw_status.get("ts")) if isinstance(raw_status, dict) else None
            if ts_val is not None and (time.time() - ts_val) < 10.0:
                recent = True
        except Exception:
            recent = False
        if not t or (hasattr(t, "is_alive") and not t.is_alive()):
            if not recent:
                stage = "idle"
    except Exception:
        stage = "idle"
    # Build response, preserving progress metadata when present
    resp: Dict[str, Any] = {
        "space": space,
        "stage": stage,
        "plan": plan,
        "delta": delta,
    }
    # Pass through progress fields from raw status
    progress = raw_status.get("progress") if isinstance(raw_status, dict) else None
    if isinstance(progress, dict):
        resp["progress"] = {"total": int(progress.get("total") or 0), "done": int(progress.get("done") or 0)}
        if progress.get("failed") is not None:
            try:
                resp["progress"]["failed"] = int(progress.get("failed") or 0)
            except Exception:
                pass
    if isinstance(raw_status, dict):
        if raw_status.get("started_at") is not None:
            try:
                started_at = float(raw_status.get("started_at"))
                resp["started_at"] = started_at
                resp["elapsed_sec"] = max(0.0, time.time() - started_at)
            except Exception:
                pass
        if raw_status.get("elapsed_sec") is not None:
            try:
                resp["elapsed_sec"] = float(raw_status.get("elapsed_sec"))
            except Exception:
                pass
        if raw_status.get("eta_sec") is not None:
            try:
                resp["eta_sec"] = float(raw_status.get("eta_sec"))
            except Exception:
                pass
        if raw_status.get("phase"):
            resp["phase"] = raw_status.get("phase")
    # Derive ETA when possible and not provided
    if resp.get("stage") == "evaluate" and "eta_sec" not in resp:
        try:
            prog = resp.get("progress") or {}
            done = int(prog.get("done") or 0)
            total = int(prog.get("total") or 0)
            elapsed = float(resp.get("elapsed_sec") or 0.0)
            if done > 0 and total > done and elapsed > 0:
                eta_val = (total - done) * (elapsed / done)
                resp["eta_sec"] = max(0.0, float(eta_val))
        except Exception:
            pass
    return resp



if __name__ == "__main__":
    try:
        uvicorn.run(
            "server_api:app",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8001)),
            log_level="info")
    except Exception as e:
        logging.error("An error occurred", exc_info=True)
        print(f"Exception with force stop: {e}", file=sys.stderr)
        #parsing_server_global.stop()
