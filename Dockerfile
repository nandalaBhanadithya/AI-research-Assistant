FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-ml.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads data/chroma data/sqlite data/classifier/model

# Set environment variables for cloud deployment
ENV EMBEDDING_PROVIDER=huggingface
ENV GENERATION_PROVIDER=groq
ENV APP_ENV=production
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
