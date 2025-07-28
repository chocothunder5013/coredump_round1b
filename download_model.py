# This script is executed during the Docker build process to download
# the required sentence-transformer model.

from sentence_transformers import SentenceTransformer
import config

# The name of the model specified in the approach.
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

print(f"Downloading model: {MODEL_NAME}...")

# Instantiate the model, which triggers the download from Hugging Face Hub.
model = SentenceTransformer(MODEL_NAME)

# Save the downloaded model files to the path specified in the config.
# This allows the main application to load it from a local path.
model.save(config.LOCAL_MODEL_PATH)

print(f"Model saved to {config.LOCAL_MODEL_PATH}")
