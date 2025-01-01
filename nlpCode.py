from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import re
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)
current_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-uncased")

# Freeze BERT parameters
for param in bert.parameters():
    param.requires_grad = False

# Define the BERT architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 4)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        a = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(a)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initialize model
model = BERT_Arch(bert)
path = os.path.join(current_dir, "sources","Bert.pt")
#path = "./sources/Bert.pt"
model.load_state_dict(torch.load(path, weights_only=True))
# Load stop words
file_path = os.path.join(current_dir, "sources","stop.tr.turkish-lucene.txt")
#file_path = r"./sources/stop.tr.turkish-lucene.txt"
with open(file_path, "r", encoding="utf-8") as file:
    stop_words_list = file.read().splitlines()

# Function to clean the text
def temizle(text):
    boslukolmadan = re.compile(r"(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})|(‘)|(“)|(”)|(°)|(\')")
    boslukile = re.compile(r"(<br\s/><br\s/?)|(-)|(/)|(:)")
    html_pattern = r'<.*?>'
    tmpL = boslukolmadan.sub("", text.lower())
    tmpL = boslukile.sub(" ", tmpL)
    tmpL = re.sub(html_pattern, "", tmpL)
    tmpL = tmpL.split()
    tmpL = [word for word in tmpL if word not in stop_words_list]
    return ' '.join(tmpL)

@app.get("/test")
def testString(testText: str): 
    testText = testText.encode('utf-8').decode('utf-8')
    testMetin = temizle(testText)
    MAX_LENGHT = 15
    tokens_unseen = tokenizer.batch_encode_plus(
        [testMetin],  # وضع النص داخل قائمة
        max_length=MAX_LENGHT,
        pad_to_max_length=True,
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        preds = model(tokens_unseen['input_ids'], tokens_unseen['attention_mask'])
        preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    pred_to_text = {
        0: "Ekonomi",
        1: "sağlık",
        2: "spor",
        3: "yaşam"
    }
    return pred_to_text.get(preds[0], "Unknown")
