FROM python:3.10-slim

LABEL maintainer="scalermetahackathon"
LABEL description="OpenEnv Earthquake Rescue Environment"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TASK_ID=easy

HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:7860/health || exit 1

CMD ["sh", "-c", "uvicorn api.app:app --app-dir /app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
