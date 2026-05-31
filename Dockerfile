FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LODE_DATA_DIR=/data

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
RUN python -m pip install --no-cache-dir '.[kuzu]'

EXPOSE 7979
CMD ["loded", "--host", "0.0.0.0", "--port", "7979"]

