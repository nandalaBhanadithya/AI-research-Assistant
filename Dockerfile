FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (minimal for Render free tier)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (skip ML requirements for free tier to save memory)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads data/chroma data/sqlite data/classifier/model

# Set environment variables for cloud deployment
ENV EMBEDDING_PROVIDER=huggingface
ENV GENERATION_PROVIDER=groq
ENV APP_ENV=production
ENV LOG_LEVEL=INFO

# Expose port (Render uses port 80 by default)
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Run the application with limited workers for memory constraints
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
