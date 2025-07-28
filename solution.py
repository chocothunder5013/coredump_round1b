import os
import json
import sys
import logging
import fitz  # PyMuPDF
import re
from collections import Counter
from datetime import datetime
from multiprocessing import Pool, cpu_count
import config
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# --- Logging Configuration ---
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NLTK Data Setup ---
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    logger.error(
        "NLTK data not found. Ensure it is downloaded during the Docker build."
    )
    sys.exit(1)

# --- PyTorch Thread Management ---
torch.set_num_threads(max(1, cpu_count() - 1))

# --- Constants ---
ROOT_DATA_PATH = os.getenv("ROOT_DATA_PATH", "/app/data")
INPUT_JSON_NAME = "challenge1b_input.json"
OUTPUT_JSON_NAME = "challenge1b_output.json"
PDF_SUBDIR_NAME = "PDFs"
TITLE_WEIGHT = 5

# --- NLP Utilities & Model Loading ---
try:
    model = SentenceTransformer(config.LOCAL_MODEL_PATH)
    STEMMER = PorterStemmer()
    STOP_WORDS = set(stopwords.words("english"))
    HEADING_PATTERN = re.compile(
        r"^((\d+\.)+\d*|\d+\.|\([a-z]\)|[A-Z]\.)\s", re.IGNORECASE
    )
    logger.info("Model and NLP utilities loaded successfully.")
except Exception:
    logger.exception("A critical error occurred during model initialization.")
    sys.exit(1)


def _detect_headers_footers(doc):
    headers_footers = set()
    page_count = len(doc)
    if page_count < 3:
        return headers_footers
    candidate_texts = {}
    for page in doc:
        page_height = page.rect.height
        top_rect = fitz.Rect(0, 0, page.rect.width, page_height * 0.15)
        bottom_rect = fitz.Rect(0, page_height * 0.85, page.rect.width, page_height)
        for text in (
            page.get_textbox(top_rect).strip(),
            page.get_textbox(bottom_rect).strip(),
        ):
            if text:
                candidate_texts[text] = candidate_texts.get(text, 0) + 1
    for text, count in candidate_texts.items():
        if count > page_count / 2:
            headers_footers.add(text)
    return headers_footers


def normalize_scores(scores):
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    return (
        [0.5] * len(scores)
        if min_s == max_s
        else [(s - min_s) / (max_s - min_s) for s in scores]
    )


def extract_sections_from_pdf(pdf_path):
    sections, doc = [], None
    try:
        doc = fitz.open(pdf_path)
        headers_footers = _detect_headers_footers(doc)
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_FONT)["blocks"]
            if not blocks:
                continue
            font_sizes = [
                s["size"]
                for b in blocks
                for l in b.get("lines", [])
                for s in l.get("spans", [])
                if s.get("text", "").strip()
            ]
            if not font_sizes:
                continue
            body_size = Counter(font_sizes).most_common(1)[0][0]
            current_title, content_buffer = f"Page {page_num + 1} Content", ""
            for block in blocks:
                block_text = "".join(
                    s["text"]
                    for l in block.get("lines", [])
                    for s in l.get("spans", [])
                ).strip()
                if block_text in headers_footers:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue
                    span = spans[0]
                    is_style = (span["flags"] & 16) or (
                        span["size"] > body_size + config.HEADING_FONT_SIZE_THRESHOLD
                    )
                    is_structural = HEADING_PATTERN.match(text) is not None
                    is_heading = (is_style or is_structural) and len(text.split()) < 12
                    if is_heading:
                        if content_buffer.strip():
                            sections.append(
                                {
                                    "document": os.path.basename(pdf_path),
                                    "section_title": current_title,
                                    "page_number": page_num + 1,
                                    "content": content_buffer.strip(),
                                }
                            )
                        current_title, content_buffer = text, ""
                    else:
                        content_buffer += text + "\n"
            if content_buffer.strip():
                sections.append(
                    {
                        "document": os.path.basename(pdf_path),
                        "section_title": current_title,
                        "page_number": page_num + 1,
                        "content": content_buffer.strip(),
                    }
                )
    except Exception:
        logger.exception(f"Failed to parse PDF: {pdf_path}")
    finally:
        if doc:
            doc.close()
    return sections


def get_refined_text_semantic_optimized(content, query_embedding):
    try:
        sentences = nltk.sent_tokenize(content)
    except Exception:
        sentences = content.split(".")

    if len(sentences) <= config.SENTENCES_PER_CHUNK:
        return " ".join(sentences)

    # FIX: Corrected the range logic to ensure all sentences are included in chunks.
    step = max(1, config.SENTENCES_PER_CHUNK - config.CHUNK_OVERLAP)
    chunks = [
        " ".join(sentences[i : i + config.SENTENCES_PER_CHUNK])
        for i in range(0, len(sentences), step)
    ]

    if not chunks:
        return " ".join(sentences[: config.SENTENCES_PER_CHUNK])

    embeddings = model.encode(chunks, convert_to_tensor=True, batch_size=32)
    return chunks[util.pytorch_cos_sim(query_embedding, embeddings)[0].argmax()]


def extract_keywords_from_persona(persona, job_to_be_done):
    text = f"{persona.get('role', '')} {job_to_be_done.get('task', '')}".lower()
    words = re.findall(r"\b\w{3,}\b", text)
    return list(set(STEMMER.stem(word) for word in words if word not in STOP_WORDS))


def analyze_documents(docs, persona, job, pdf_dir):
    pdf_paths = [
        os.path.join(pdf_dir, doc["filename"])
        for doc in docs
        if os.path.isfile(os.path.join(pdf_dir, doc["filename"]))
    ]
    if not pdf_paths:
        return [], []

    keywords = extract_keywords_from_persona(persona, job)
    keyword_pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, keywords)) + r")\b", re.IGNORECASE
    )

    candidate_sections = []
    with Pool(min(cpu_count(), len(pdf_paths))) as pool:
        for sections in pool.imap_unordered(extract_sections_from_pdf, pdf_paths):
            for section in sections:
                if keyword_pattern.search(
                    section["section_title"]
                ) or keyword_pattern.search(section["content"]):
                    candidate_sections.append(section)

    if not candidate_sections:
        return [], []

    query_embedding = model.encode(job.get("task", ""), convert_to_tensor=True)
    texts_for_embedding = [
        f"{s['section_title']}. {' '.join(s['content'].split()[:250])}"
        for s in candidate_sections
    ]
    embeddings = model.encode(
        texts_for_embedding, convert_to_tensor=True, batch_size=128
    )
    semantic_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0].tolist()

    keyword_scores = []
    for section in candidate_sections:
        title_matches = (
            len(keyword_pattern.findall(section["section_title"])) * TITLE_WEIGHT
        )
        content_matches = len(keyword_pattern.findall(section["content"]))
        word_count = len(section["content"].split())
        keyword_scores.append(
            title_matches
            + (content_matches / word_count * 100 if word_count > 0 else 0)
        )

    norm_sem = normalize_scores(semantic_scores)
    norm_key = normalize_scores(keyword_scores)

    for i, section in enumerate(candidate_sections):
        section["score"] = (norm_key[i] * config.KEYWORD_SCORE_WEIGHT) + (
            norm_sem[i] * config.SEMANTIC_SCORE_WEIGHT
        )

    top_sections = sorted(candidate_sections, key=lambda x: x["score"], reverse=True)[
        : config.TOP_N_SECTIONS
    ]

    for rank, section in enumerate(top_sections, 1):
        section["importance_rank"] = rank

    extracted = [
        {k: v for k, v in s.items() if k not in ("score", "content")}
        for s in top_sections
    ]
    analysis = [
        {
            "document": s["document"],
            "refined_text": get_refined_text_semantic_optimized(
                s["content"], query_embedding
            ),
            "page_number": s["page_number"],
        }
        for s in top_sections
    ]
    return extracted, analysis


def generate_final_output(data, extracted, analysis):
    return {
        "metadata": {
            "input_documents": [
                d.get("filename", "N/A") for d in data.get("documents", [])
            ],
            "persona": data.get("persona", {}).get("role", "N/A"),
            "job_to_be_done": data.get("job_to_be_done", {}).get("task", "N/A"),
            "processing_timestamp": datetime.now().isoformat(),
        },
        "extracted_sections": extracted,
        "subsection_analysis": analysis,
    }


def process_collection(collection_path):
    input_json = os.path.join(collection_path, INPUT_JSON_NAME)
    pdf_dir = os.path.join(collection_path, PDF_SUBDIR_NAME)
    output_json = os.path.join(collection_path, OUTPUT_JSON_NAME)

    if not (os.path.isfile(input_json) and os.path.isdir(pdf_dir)):
        logger.warning(f"Skipping collection '{collection_path}': Missing inputs.")
        return

    try:
        with open(input_json, "r") as f:
            data = json.load(f)
        if not all(k in data for k in ("documents", "persona", "job_to_be_done")):
            logger.error(f"Invalid JSON in {input_json}.")
            return
        extracted, analysis = analyze_documents(
            data["documents"], data["persona"], data["job_to_be_done"], pdf_dir
        )
        with open(output_json, "w") as f:
            json.dump(generate_final_output(data, extracted, analysis), f, indent=4)
        logger.info(f"Successfully processed collection: {collection_path}")
    except Exception:
        logger.exception(f"Error processing collection: {collection_path}")


def main():
    if not os.path.isdir(ROOT_DATA_PATH):
        logger.error(f"Root data path not found: {ROOT_DATA_PATH}")
        return
    collections = [
        os.path.join(ROOT_DATA_PATH, d)
        for d in os.listdir(ROOT_DATA_PATH)
        if os.path.isdir(os.path.join(ROOT_DATA_PATH, d))
    ]
    if not collections:
        logger.warning("No data collections found.")
        return
    for collection in sorted(collections):
        process_collection(collection)


if __name__ == "__main__":
    main()
