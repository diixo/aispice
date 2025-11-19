import requests
import spacy
import json
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup


# 1. --- Эталон ASPICE MAN.3 Base Practices ---
ASPICE_MAN3_BP = [
    {"BP": "BP1", "action": ["establish"], "object": ["project plan"]},
    {"BP": "BP2", "action": ["identify"], "object": ["tasks", "dependencies", "resources"]},
    {"BP": "BP3", "action": ["define"], "object": ["schedule", "milestones", "deliverables"]},
    {"BP": "BP4", "action": ["review", "update"], "object": ["project plan"]},
    {"BP": "BP5", "action": ["allocate", "monitor"], "object": ["responsibilities", "progress"]}
]


def extract_confluence_text_from_file(file_path):
    """Извлекает читаемый текст из локального HTML-файла Confluence."""
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Удалим скрипты, стили и ненужное
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Извлечём только текст
    text = soup.get_text(separator="\n")

    # Уберём лишние пробелы и пустые строки
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)

    return cleaned


# 3. --- Очистка и сегментация ---
def segment_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]


# 4. --- Простая имитация аннотации слотами (заглушка) ---
# В реальности здесь вызывается fine-tuned LLM.
def annotate_slots(sentence):
    sentence_lower = sentence.lower()
    slots = {
        "intention": "define",
        "action": [],
        "relation": "inherent",
        "subject": "process",
        "object": [],
        "context": []
    }
    # Простейшая эвристика по ключевым словам
    for bp in ASPICE_MAN3_BP:
        for act in bp["action"]:
            if act in sentence_lower:
                slots["action"].append(act)
        for obj in bp["object"]:
            if obj in sentence_lower:
                slots["object"].append(obj)
    return slots

# 5. --- Семантическое сравнение ---
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def match_with_bp(sentence, slots):
    results = []
    for bp in ASPICE_MAN3_BP:
        score = 0
        for act in bp["action"]:
            act_score = util.cos_sim(model.encode(sentence), model.encode(act)).item()
            score = max(score, act_score)
        for obj in bp["object"]:
            obj_score = util.cos_sim(model.encode(sentence), model.encode(obj)).item()
            score = max(score, obj_score)
        results.append({"BP": bp["BP"], "score": score})
    return sorted(results, key=lambda x: x["score"], reverse=True)[0]


# 6. --- Основной пайплайн ---
def evaluate_man3(text):
    sentences = segment_text(text)
    print(len(sentences))
    report = {"MAN.3": {"coverage": 0, "details": []}}
    covered_bp = set()

    for sent in sentences:
        slots = annotate_slots(sent)
        best_match = match_with_bp(sent, slots)
        if best_match["score"] > 0.45:  # порог близости
            report["MAN.3"]["details"].append({
                "sentence": sent,
                "best_BP": best_match["BP"],
                "score": round(best_match["score"], 2)
            })
            covered_bp.add(best_match["BP"])

    report["MAN.3"]["coverage"] = len(covered_bp) / len(ASPICE_MAN3_BP)
    return report


#######################################################################

confluence_text = """
The project plan is established at the beginning of the project.
Tasks and dependencies are identified using the WBS.
The plan is reviewed and updated monthly.
Resources and responsibilities are allocated to each milestone.
"""


txt = extract_confluence_text_from_file("corpus/page911478017.html")

result = evaluate_man3(txt)
print(json.dumps(result, indent=2))
