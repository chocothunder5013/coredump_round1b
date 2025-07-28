# ---- Build Stage ----
# Use a full Python image to build dependencies and download assets.
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies required for building Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to isolate dependencies.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# --- Asset Automation ---
# Pre-download NLTK data and the ML model to ensure offline availability.
# FIX: Copy config.py so the download script can import it.
COPY download_model.py config.py .
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
RUN python download_model.py


# ---- Final Stage ----
# Use a minimal, clean base image for the production application.
FROM python:3.10-slim

WORKDIR /app

# Copy the isolated virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy all pre-downloaded assets (ML model and NLTK data).
COPY --from=builder /app/model ./model
COPY --from=builder /root/nltk_data /root/nltk_data

# Copy the application source code.
COPY solution.py .
COPY config.py .

# Set environment variables for the runtime environment.
ENV PATH="/opt/venv/bin:$PATH"
ENV NLTK_DATA="/root/nltk_data"
ENV ROOT_DATA_PATH="/app/data"

# Define the entrypoint to run the solution script.
ENTRYPOINT ["python", "solution.py"]