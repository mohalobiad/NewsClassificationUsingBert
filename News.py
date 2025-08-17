import os
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import re 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(current_dir, "sources","downloadedBertModel")
path = os.path.join(current_dir, "sources","outputBert.pt")
stopwords_path = os.path.join(current_dir, "sources", "english_stopwords.txt") 
index_path = os.path.join(current_dir, "index.html")


tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
bert = DistilBertModel.from_pretrained(MODEL_DIR, local_files_only=True)
class DistilBERT_Arch(nn.Module):
    def __init__(self, bert, num_labels=5, dropout=0.1):
        super().__init__()
        self.bert = bert
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(h, 512)
        self.fc2 = nn.Linear(512, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sent_emb = self.mean_pool(out.last_hidden_state, attention_mask) 
        x = self.fc1(sent_emb); x = self.relu(x); x = self.dropout(x)
        x = self.fc2(x)
        return self.logsoftmax(x)
model = DistilBERT_Arch(bert, num_labels=5)
model.load_state_dict(torch.load(path))

with open(stopwords_path, "r", encoding="utf-8") as f:
    stop_words = set(w.strip() for w in f if w.strip())
def clean(line: str) -> str:
    punctuation_no_space = re.compile(r"(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})|(‘)|(“)|(”)|(°)|(\')")
    punctuation_with_space = re.compile(r"(<br\s*/?><br\s*/?>)|(-)|(/)|(:)")
    html_tags = r'<.*?>'

    s = line.lower()
    s = punctuation_no_space.sub("", s)
    s = punctuation_with_space.sub(" ", s)
    s = re.sub(html_tags, "", s)
    tokens = s.split()
    tokens = [w for w in tokens if w not in stop_words]   
    return " ".join(tokens)

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(index_path)
@app.get("/test")
def testString(testText: str): 
    cleaned_text = clean(testText.encode('utf-8').decode('utf-8'))

    MAX_LENGHT = 13
    tokens_unseen = tokenizer.batch_encode_plus(
        [cleaned_text],  
        max_length=MAX_LENGHT,
        padding='max_length',   
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        preds = model(tokens_unseen['input_ids'], tokens_unseen['attention_mask'])
        preds = preds.detach().cpu().numpy()

    preds = np.argmax(preds, axis=1)

    pred_to_text = {
        0: "POLITICS",
        1: "WELLNESS",
        2: "ENTERTAINMENT",
        3: "TRAVEL",
        4: "STYLE & BEAUTY"
    }
    return pred_to_text.get(preds[0], "Unknown")

