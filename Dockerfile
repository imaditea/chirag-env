cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app
COPY . /app

ENV PYTHONPATH=/app

RUN pip install numpy gymnasium pyyaml

CMD ["python", "chirag_env.py"]
EOF
