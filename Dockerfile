FROM python:3.10-slim

WORKDIR /app
COPY . /app

ENV PYTHONPATH=/app

RUN pip install numpy gymnasium pyyaml fastapi uvicorn requests

CMD ["python", "server/app.py"]
