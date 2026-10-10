FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LABTRACK_DB=/app/data/labtrack.db
WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt
COPY app.py ./
COPY labtrack/ ./labtrack/
COPY scripts/ ./scripts/
COPY data/ ./data/
RUN groupadd --gid 10001 labtrack \
    && useradd --uid 10001 --gid labtrack --create-home labtrack \
    && chown -R labtrack:labtrack /app/data
USER labtrack

VOLUME ["/app/data"]
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
