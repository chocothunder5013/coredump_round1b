import os
import json
import sys
import logging
import fitz  # PyMuPDF
import re
from collections import Counter, defaultdict
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
# Performance: Control threads to prevent over-subscription on CPU.
torch.set_num_threads(config.PYTORCH_NUM_THREADS)

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
    # Sample first, middle, and last pages for efficiency in very long docs
    pages_to_check = [doc[0], doc[page_count // 2], doc[-1]]
    for page in pages_to_check:
        page_height = page.rect.height
        # Check 10% margins
        top_rect = fitz.Rect(0, 0, page.rect.width, page_height * 0.10)
        bottom_rect = fitz.Rect(0, page_height * 0.90, page.rect.width, page_height)
        for text in (
            page.get_textbox(top_rect).strip(),
            page.get_textbox(bottom_rect).strip(),
        ):
            if text and len(text) > 5:
                candidate_texts[text] = candidate_texts.get(text, 0) + 1
    # Identify text that appears on more than half the checked pages
    for text, count in candidate_texts.items():
        if count > len(pages_to_check) / 2:
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
                is_header_or_footer = False
                block_text_unprocessed = "".join(
                    s["text"]
                    for l in block.get("lines", [])
                    for s in l.get("spans", [])
                ).strip()

                if not block_text_unprocessed:
                    continue

                # Check if the whole block text is a header/footer
                for hf in headers_footers:
                    if hf in block_text_unprocessed:
                        is_header_or_footer = True
                        break
                if is_header_or_footer:
                    continue

                block_text = ""
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue

                    span = spans[0]
                    is_style = (span["flags"] & 16) or (
                        span["size"] > body_size * config.HEADING_FONT_SIZE_THRESHOLD
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


def get_refined_text_batch(top_sections, query_embedding):
    """
    Performance: Processes all top sections in a single batch for efficiency.
    """
    all_chunks = []
    section_chunk_map = defaultdict(list)

    for i, section in enumerate(top_sections):
        content = section["content"]
        try:
            sentences = nltk.sent_tokenize(content)
        except Exception:
            sentences = content.split(".")

        if len(sentences) <= config.SENTENCES_PER_CHUNK:
            section_chunk_map[i].append(" ".join(sentences))
            continue

        step = max(1, config.SENTENCES_PER_CHUNK - config.CHUNK_OVERLAP)
        chunks = [
            " ".join(sentences[j : j + config.SENTENCES_PER_CHUNK])
            for j in range(0, len(sentences), step)
        ]
        if chunks:
            section_chunk_map[i].extend(chunks)

    # Flatten all chunks from all sections into one list for batch encoding
    master_chunk_list = [
        chunk for chunks in section_chunk_map.values() for chunk in chunks
    ]

    if not master_chunk_list:
        return ["No relevant text found." for _ in top_sections]

    # Perform one single, large batch encoding operation
    chunk_embeddings = model.encode(
        master_chunk_list, convert_to_tensor=True, batch_size=128
    )

    # Calculate similarities and find the best chunk for each section
    refined_texts = []
    chunk_offset = 0
    for i in range(len(top_sections)):
        num_chunks = len(section_chunk_map[i])
        if num_chunks == 0:
            refined_texts.append("Content too short to refine.")
            continue

        # Get the slice of embeddings corresponding to the current section's chunks
        section_embeddings = chunk_embeddings[chunk_offset : chunk_offset + num_chunks]

        # Find the best chunk
        cos_scores = util.pytorch_cos_sim(query_embedding, section_embeddings)[0]
        best_chunk_index = cos_scores.argmax()

        refined_texts.append(section_chunk_map[i][best_chunk_index])
        chunk_offset += num_chunks

    return refined_texts


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
    # Performance: Pre-compile regex for faster matching.
    keyword_pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, keywords)) + r")\b", re.IGNORECASE
    )

    candidate_sections = []
    # Performance: Use multiprocessing for I/O-bound PDF parsing.
    with Pool(min(cpu_count(), len(pdf_paths))) as pool:
        # Use imap_unordered for memory efficiency
        for sections in pool.imap_unordered(extract_sections_from_pdf, pdf_paths):
            for section in sections:
                # Performance: Pre-filter sections to reduce semantic search space.
                if keyword_pattern.search(
                    section["section_title"]
                ) or keyword_pattern.search(section["content"]):
                    candidate_sections.append(section)

    if not candidate_sections:
        return [], []

    query_embedding = model.encode(job.get("task", ""), convert_to_tensor=True)
    # Performance: Create text for embedding as a generator to save memory.
    texts_for_embedding = (
        f"{s['section_title']}. {' '.join(s['content'].split()[:250])}"
        for s in candidate_sections
    )
    # Performance: Encode all candidate sections in a single batch operation.
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

    if not top_sections:
        return [], []

    for rank, section in enumerate(top_sections, 1):
        section["importance_rank"] = rank

    # Performance: Generate all refined texts in a single batch operation.
    refined_texts_list = get_refined_text_batch(top_sections, query_embedding)

    extracted = [
        {k: v for k, v in s.items() if k not in ("score", "content")}
        for s in top_sections
    ]
    analysis = [
        {
            "document": s["document"],
            "refined_text": refined_texts_list[i],
            "page_number": s["page_number"],
        }
        for i, s in enumerate(top_sections)
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

        logger.info(f"Analyzing documents for collection: {collection_path}")
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
