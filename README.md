# 🧠 Persona-Driven Document Intelligence

[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10-green?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Only-red?logo=pytorch)](https://pytorch.org/)


An **offline, CPU-optimized** intelligent document analysis engine that extracts the most relevant sections from PDF collections based on user personas and specific tasks. Built for resource-constrained environments with zero network dependency at runtime.

## Key Features

- **Fully Offline**: Complete self-contained Docker image with no network requirements
- **High Performance**: Parallel PDF processing using Python multiprocessing
- **Hybrid Intelligence**: Combines keyword density + semantic similarity scoring
- **Persona-Driven**: Dynamically adapts analysis based on user context and tasks
- **Smart Document Processing**: Advanced header/footer detection and section identification
- **Docker Ready**: One-command build and deployment

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Collection│    │  Analysis Engine │    │ Ranked Results  │
│                 │───▶│                  │───▶│                 │
│ • Multiple PDFs │    │ • Text Extraction│    │ • Top 5 Sections│
│ • Input Config  │    │ • Hybrid Scoring │    │ • Relevance Score│
│ • User Persona  │    │ • ML Processing  │    │ • Smart Summary │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

1. **Parallel PDF Processor**: Extracts and structures content from multiple PDFs simultaneously
2. **Persona Analyzer**: Dynamically extracts keywords from user persona and job descriptions
3. **Hybrid Scoring Engine**: 
   - **Keyword Density** (40%): Frequency-based matching with stemming
   - **Semantic Similarity** (60%): ML-powered contextual understanding
4. **Smart Summarizer**: Context-aware text refinement using sentence transformers

## Prerequisites

- **Docker Desktop** (with Linux containers enabled)
- **8GB+ RAM** recommended for optimal performance
- **2GB+ storage** for Docker image and models

## Quick Start

### 1. Build the Docker Image

The build process automatically downloads all required dependencies and ML models:

```bash
docker build --platform linux/amd64 -t coredump:round1b .
```

> **Build Time**: ~10-15 minutes (depending on internet speed)

### 2. Run the Analysis

Process all document collections in your current directory:

```bash
docker run --rm -v "$(pwd)":/app/data --network none coredump:round1b
```

> **Note**: `--network none` ensures complete offline operation

## Project Structure

```
coredump_round1b/
├── solution.py              # Main analysis engine
├── config.py               # Configuration parameters
├── Dockerfile               # Multi-stage container build
├── requirements.txt         # Python dependencies
├── download_model.py        # ML model downloader
├── approach_explanation.md  # Technical methodology
├── README.md               # This file
└── Collection [1-3]/       # Sample document collections
    ├── PDFs/               # Source PDF documents
    ├── challenge1b_input.json   # Analysis configuration
    └── challenge1b_output.json  # Generated results
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KEYWORD_SCORE_WEIGHT` | 0.4 | Weight for keyword density scoring |
| `SEMANTIC_SCORE_WEIGHT` | 0.6 | Weight for semantic similarity scoring |
| `TOP_N_SECTIONS` | 5 | Number of top sections to return |
| `HEADING_FONT_SIZE_THRESHOLD` | 1.5 | Font size ratio for heading detection |

## Input/Output Format

### Input Structure (`challenge1b_input.json`)
```json
{
    "challenge_info": {
        "challenge_id": "round_1b_002",
        "test_case_name": "travel_planner",
        "description": "France Travel"
    },
    "documents": [
        {
            "filename": "document.pdf",
            "title": "Document Title"
        }
    ],
    "persona": "Travel enthusiast planning a vacation",
    "job_to_be_done": "Find the best restaurants in South of France"
}
```

### Output Structure (`challenge1b_output.json`)
```json
{
    "sections": [
        {
            "source_document": "document.pdf",
            "section_header": "Restaurant Recommendations",
            "content": "Refined section content...",
            "relevance_score": 0.87
        }
    ]
}
```

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Runtime** | Python 3.10 | Core application |
| **ML Framework** | PyTorch (CPU) | Neural network inference |
| **NLP Model** | multi-qa-MiniLM-L6-cos-v1 | Semantic similarity |
| **PDF Processing** | PyMuPDF (fitz) | Document parsing |
| **Text Processing** | NLTK | Tokenization & stemming |
| **Containerization** | Docker | Deployment & isolation |

## Algorithm Details

### 1. Document Processing
- **Header/Footer Detection**: Statistical analysis of recurring elements
- **Section Identification**: Font-based heuristics + pattern matching
- **Content Extraction**: Text + metadata preservation

### 2. Scoring Methodology
```python
final_score = (keyword_score * 0.4) + (semantic_score * 0.6)
```

- **Keyword Score**: TF-IDF with stemming and stop-word filtering
- **Semantic Score**: Cosine similarity between embeddings
- **Normalization**: Both scores scaled to [0,1] range

### 3. Summary Generation
- **Chunking**: Overlapping sentence groups (4 sentences + 1 overlap)
- **Selection**: Most semantically relevant chunk per section
- **Refinement**: Context-aware text optimization

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Processing Speed** | ~2-3 PDFs/second |
| **Memory Usage** | ~1.5GB peak |
| **Model Size** | ~80MB |
| **Docker Image** | ~2.1GB |
| **CPU Utilization** | Multi-core optimized |

## Security & Privacy

- **Zero Network Dependency**: Complete offline operation
- **No Data Persistence**: Documents processed in memory only
- **Containerized**: Isolated execution environment
- **No External APIs**: All processing done locally

## 🔍 Troubleshooting

### Common Issues

**Docker Build Fails**
```bash
# Ensure Docker Desktop is running
docker --version
```

**Memory Issues**
```bash
# Increase Docker memory allocation to 4GB+
# Docker Desktop → Settings → Resources → Memory
```

**Permission Errors (Windows)**
```bash
# Use PowerShell as Administrator
docker run --rm -v "${PWD}:/app/data" --network none coredump:round1b
```

**Tokenizer Warnings**
```bash
# These are informational and don't affect functionality
# To suppress: set TOKENIZERS_PARALLELISM=false
```

## Advanced Usage

### Custom Configuration
```bash
# Mount custom config
docker run --rm \
  -v "$(pwd)":/app/data \
  -v "$(pwd)/custom_config.py":/app/config.py \
  --network none coredump:round1b
```

### Batch Processing
```bash
# Process multiple directories
for dir in Collection_*; do
  docker run --rm -v "$(pwd)/$dir":/app/data --network none coredump:round1b
done
```
