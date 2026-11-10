FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R app:app /app /home/app
USER app

EXPOSE 8501
CMD ["sh", "-c", "exec streamlit run src/demo_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true --browser.gatherUsageStats=false"]
