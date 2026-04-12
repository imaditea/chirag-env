FROM python:3.10-slim

WORKDIR /app
COPY . /app

ENV PYTHONPATH=/app

RUN pip install numpy gymnasium pyyaml fastapi uvicorn requests

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
