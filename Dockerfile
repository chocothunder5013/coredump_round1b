# ---- Build Stage ----
# Use a specific slim image to ensure build reproducibility and a smaller base.
FROM python:3.10-slim-bullseye as builder

WORKDIR /app

# Add a non-root user for security.
# This prevents build and run processes from having root privileges.
RUN useradd --create-home appuser
RUN chown -R appuser:appuser /app

# Install system dependencies required for building Python packages.
# Combined with cleaning steps to reduce layer size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to isolate dependencies.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies from the pinned requirements file.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# --- Asset Automation ---
# Pre-download NLTK data and the ML model to ensure offline availability.
COPY --chown=appuser:appuser download_model.py config.py .
# Combine NLTK download and model download into a single RUN layer.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" && \
    python download_model.py

# --- SIZE OPTIMIZATION (NEW) ---
# Aggressively prune the virtual environment to reduce image size.
# This removes tests, documentation, and compilation artifacts from installed packages.
RUN find /opt/venv -follow -type f -name '*.a' -delete \
    && find /opt/venv -follow -type f -name '*.pyc' -delete \
    && find /opt/venv -follow -type d -name '__pycache__' -delete \
    && find /opt/venv -follow -type d -name 'tests' -delete \
    && find /opt/venv -follow -type d -name 'test' -delete \
    && rm -rf /root/.cache

USER appuser

# ---- Final Stage ----
# Use the same minimal base image for the production application.
FROM python:3.10-slim-bullseye

WORKDIR /app

# Add the non-root user to the final image.
RUN useradd --create-home appuser

# Copy the isolated virtual environment from the builder stage.
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy all pre-downloaded assets (ML model and NLTK data).
COPY --from=builder --chown=appuser:appuser /app/model ./model
COPY --from=builder --chown=appuser:appuser /home/appuser/nltk_data /home/appuser/nltk_data

# Copy the application source code.
COPY --chown=appuser:appuser solution.py .
COPY --chown=appuser:appuser config.py .

# Set environment variables for the runtime environment.
ENV PATH="/opt/venv/bin:$PATH"
ENV NLTK_DATA="/home/appuser/nltk_data"
ENV ROOT_DATA_PATH="/app/data"

# Switch to the non-root user for execution.
USER appuser

# Define the entrypoint to run the solution script.
ENTRYPOINT ["python", "solution.py"]