
import requests
from bs4 import BeautifulSoup
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import List, Dict, Any
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")  # sentence segmentation, optional NER

# --- CONFIG ------------------------------------------------
CONFLUENCE_BASE = "https://luxproject.luxoft.com/confluence/display/SWT1AQ/"
CONFLUENCE_AUTH = ("user", "token")  # or use bearer token
EMBED_MODEL_NAME = "all-mpnet-base-v2"  # or another embedding model
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)


# If you have an LLM for slot annotation:
# - Could be an API (OpenAI) or a local model. Here we provide two hooks:
def llm_annotate_prompt(sentence: str) -> str:
    # Template prompt to send to LLM
    return f"""
Annotate the following sentence into JSON with slots:
intention, action, relation, subject, object, emotion, context (if any).
Return only valid JSON.

Sentence: \"{sentence}\"
"""

def call_llm_api(prompt: str) -> Dict[str,Any]:
    # Placeholder: implement your call to an LLM and parse JSON
    # Example: call OpenAI or local LLM -> return parsed JSON
    raise NotImplementedError("Implement LLM call here")


def fetch_confluence_page(page_id: str) -> str:
    url = f"{CONFLUENCE_BASE}/{page_id}?expand=body.storage"
    r = requests.get(url, auth=CONFLUENCE_AUTH)
    r.raise_for_status()
    data = r.json()
    html = data["body"]["storage"]["value"]
    return html


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # remove scripts, styles, macros etc.
    for tag in soup(["script","style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Normalize whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text


# --- Sentence segmentation ----------------------------------
def split_to_sentences(text: str) -> List[str]:
    doc = nlp(text)
    sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sents


# --- Slot annotation (two modes) ----------------------------
def annotate_with_llm(sentence: str) -> Dict[str,Any]:
    # Use an LLM to produce the slots JSON.
    prompt = llm_annotate_prompt(sentence)
    response_json = call_llm_api(prompt)  # implement your API
    return response_json


def heuristic_annotate(sentence: str) -> Dict[str,Any]:
    # Lightweight fallback: rule-based heuristics to fill basic slots.
    s = sentence.lower()
    out = {
      "intention": None,
      "action": None,
      "relation": None,
      "subject": None,
      "object": None,
      "emotion": "neutral",
      "context": None
    }
    # examples of simple rules:
    if s.endswith("?") or s.startswith("is ") or s.startswith("are "):
        out["intention"] = "verify"
    elif any(w in s for w in ["should", "must", "shall", "required"]):
        out["intention"] = "define_rule"
    else:
        out["intention"] = "provide_info"

    # naive subject/object extraction with spaCy
    doc = nlp(sentence)
    # try to find noun chunks for subject/object
    nps = [nc.text for nc in doc.noun_chunks]
    if nps:
        out["subject"] = nps[0]
        out["object"] = nps[1] if len(nps) > 1 else None

    # find verbs
    verbs = [tok.lemma_ for tok in doc if tok.pos_ == "VERB"]
    out["action"] = verbs if verbs else None

    # quick relation guess
    if "collect" in s or "communicat" in s:
        out["relation"] = "inherent"
    elif "belong" in s or "own" in s:
        out["relation"] = "owned_by"
    else:
        out["relation"] = "related_to"
    return out


# --- Embeddings and matching --------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)


def compute_match_score(ann_embedding: np.ndarray, ref_embeddings: np.ndarray) -> float:
    # cosine similarity since we normalized
    sims = np.dot(ref_embeddings, ann_embedding)
    return float(np.max(sims))


# --- Matching pipeline: single sentence against all ASPICE refs
def match_sentence_to_aspice(annotation: Dict[str,Any], aspice_refs: List[Dict[str,Any]], ref_embeddings: np.ndarray, threshold=0.75) -> Dict[str,Any]:
    text = annotation.get("text") or " ".join([str(v) for v in annotation.get("slots", {}).values() if v])
    emb = embed_texts([text])[0]
    best_score = compute_match_score(emb, ref_embeddings)
    best_idx = int(np.argmax(np.dot(ref_embeddings, emb)))
    best_ref = aspice_refs[best_idx]
    matched = best_score >= threshold
    return {
        "text": text,
        "annotation": annotation,
        "best_ref_id": best_ref["id"],
        "best_ref_question": best_ref["question"],
        "score": best_score,
        "matched": matched
    }

# --- Document level aggregation & scoring -------------------
def aggregate_matches(matches: List[Dict[str,Any]], aspice_refs: List[Dict[str,Any]]):
    # coverage per aspice ref: count matched sentences
    coverage = {r["id"]: {"ref": r, "matches": []} for r in aspice_refs}
    for m in matches:
        if m["matched"]:
            coverage[m["best_ref_id"]]["matches"].append(m)
    # compute simple coverage score = matched_refs / total_refs
    matched_refs = sum(1 for v in coverage.values() if len(v["matches"])>0)
    total_refs = len(aspice_refs)
    return {
        "coverage_ratio": matched_refs / total_refs if total_refs>0 else 0.0,
        "coverage_details": coverage
    }


# --- Report generation --------------------------------------
def generate_report(aggregate_result):
    # Simple JSON report scaffolding
    report = {
        "summary": {
            "coverage_ratio": aggregate_result["coverage_ratio"]
        },
        "details": []
    }
    for ref_id, info in aggregate_result["coverage_details"].items():
        report["details"].append({
            "id": ref_id,
            "question": info["ref"]["question"],
            "matches_count": len(info["matches"]),
            "examples": [m["text"] for m in info["matches"][:5]]
        })
    return report


# --- Orchestrator: run pipeline on a Confluence page -----------
def run_pipeline_on_page(page_id: str, aspice_refs: List[Dict[str,Any]], use_llm=True, match_threshold=0.75):
    html = fetch_confluence_page(page_id)
    text = html_to_text(html)
    sentences = split_to_sentences(text)

    # prepare reference embeddings once
    ref_texts = [r["question"] + " " + " ".join(r.get("examples", [])) for r in aspice_refs]
    ref_embeddings = embed_texts(ref_texts)

    matches = []
    for s in tqdm(sentences, desc="Annotating sentences"):
        if use_llm:
            try:
                ann = annotate_with_llm(s)
            except Exception:
                ann = heuristic_annotate(s)
        else:
            ann = heuristic_annotate(s)
        # store text inside annotation for embedding
        ann["text"] = s
        match_info = match_sentence_to_aspice(ann, aspice_refs, ref_embeddings, threshold=match_threshold)
        matches.append(match_info)

    agg = aggregate_matches(matches, aspice_refs)
    report = generate_report(agg)
    return report, matches
