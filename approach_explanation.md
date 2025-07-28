### **Approach Explanation: Persona-Driven Document Intelligence**

Our solution tackles the "Persona-Driven Document Intelligence" challenge by implementing a sophisticated, multi-stage analysis engine. The entire process is optimized for performance, accuracy, and maintainability within the specified offline, resource-constrained environment.

#### **Methodology**

1.  **Automated Offline Asset Preparation**: During the Docker build phase, we automate the download of all necessary external assets. This includes the `multi-qa-MiniLM-L6-cos-v1` sentence-transformer model and NLTK's `punkt` and `stopwords` packages. This ensures the final container is fully self-contained and requires no network access at runtime.

2.  **Parallelized Content Extraction**: We use Python's `multiprocessing` library with a memory-efficient `imap` iterator to parse multiple PDFs in parallel. To improve accuracy, we first detect and filter out common headers and footers. Section breaks are then identified using a hybrid of font styling heuristics (e.g., bold or larger text) and structural pattern matching (e.g., numbered headings like "2.1 Topic").

3.  **Dynamic Persona Analysis**: The system analyzes the `persona` and `job_to_be_done` to dynamically extract core keywords. To improve matching, these keywords are stemmed using NLTK (e.g., "analyzing" becomes "analyz"), and a comprehensive stop-word list is used for filtering.

4.  **Hybrid Scoring Model**: This is the core of our solution. To rank the relevance of each section, we use a hybrid scoring model that combines two distinct techniques:
    * **Keyword Density Scoring**: We calculate a score based on the frequency of the stemmed keywords, giving extra weight to matches found in section titles.
    * **Semantic Similarity Scoring**: Using the pre-loaded language model, we calculate the cosine similarity between the user's "job-to-be-done" and each section's text.
    * **Score Normalization**: **Crucially, both the keyword and semantic scores are normalized to a common 0-1 scale before being combined.** This prevents any single score from unfairly dominating the ranking and makes the final weighted average more reliable and meaningful.

5.  **Optimized Semantic Summary**: For the top 5 ranked sections, we generate a context-aware summary. Instead of analyzing every sentence, we group sentences into overlapping chunks, use the language model to find the single most semantically relevant chunk, and present that as the refined text. This is significantly more performant than a sentence-by-sentence analysis.

#### **Configuration**

Key parameters like the language model path, scoring weights, and text processing settings are managed in a central `config.py` file for easy tuning and maintenance.