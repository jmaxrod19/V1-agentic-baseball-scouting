# Container for the scouting-report web app.
# Runs anywhere Docker runs; also Hugging Face Spaces-ready (non-root user,
# writable cache dirs, listens on 7860).
FROM python:3.11-slim

# HF Spaces run the container as uid 1000 — create that user and give it a
# writable HOME so matplotlib and pybaseball can write their caches.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLCONFIGDIR=/home/user/.cache/matplotlib

WORKDIR /home/user/app

# Install dependencies first so this layer is cached across code changes.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Then the application code.
COPY --chown=user:user . .

EXPOSE 7860
# Honor $PORT when the host sets one (e.g. Render); default to 7860 (HF Spaces).
CMD ["sh", "-c", "uvicorn webapp:app --host 0.0.0.0 --port ${PORT:-7860}"]
