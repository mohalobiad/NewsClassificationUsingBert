#!/bin/bash
git clone https://github.com/mohalobiad/NewsClassificationUsingBert.git /app
cd /app
pip install -r requirements.txt
uvicorn nlpCode:app --host 0.0.0.0 --port $PORT
