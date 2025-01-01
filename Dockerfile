myFROM python:3.9-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    python3-pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
