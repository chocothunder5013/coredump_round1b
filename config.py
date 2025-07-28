# --- Model & Scoring Configuration ---

# Path where the ML model is saved during the Docker build
# and loaded from during runtime.
LOCAL_MODEL_PATH = "./model"

# Weights for combining keyword and semantic scores. Must sum to 1.0.
# A higher semantic weight favors contextual relevance over direct keyword matches.
KEYWORD_SCORE_WEIGHT = 0.4
SEMANTIC_SCORE_WEIGHT = 0.6

# Number of top-ranked sections to return in the final output.
TOP_N_SECTIONS = 5

# --- Text Processing Configuration ---

# Threshold for identifying a heading based on font size.
# A line is a heading if its font size is this much larger than the body text.
HEADING_FONT_SIZE_THRESHOLD = 1.5

# Configuration for the summary refinement process.
# Text is split into chunks of this many sentences.
SENTENCES_PER_CHUNK = 4
# Number of sentences to overlap between consecutive chunks to maintain context.
CHUNK_OVERLAP = 1
