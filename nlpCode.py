from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import re
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import gdown
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)
if not os.path.exists("sources"):
    # إذا لم يكن موجودًا، نقوم بإنشائه
    os.makedirs("sources")
# Load BERT model and tokenizer
def downloadHuggingFaceBertModel():
    url = 'https://drive.google.com/uc?id=1n4OPtO5Gn7hWuiHTqD9zg4Klf6CQrYm9'
    output = 'sources/downloadedBertModel/vocab.txt'
    gdown.download(url, output, quiet=False)
    url = 'https://drive.google.com/uc?id=1eCAgS70OiVJMZtRYEKj74LNLrx-Krilg'
    output = 'sources/downloadedBertModel/tokenizer.json'
    gdown.download(url, output, quiet=False)
    url = 'https://drive.google.com/uc?id=1-tAlfkmMh1Ds859bkXo2oT3dXUchnm5s'
    output = 'sources/downloadedBertModel/tokenizer_config.json'
    gdown.download(url, output, quiet=False)
    url = 'https://drive.google.com/uc?id=1tPnfCniw16UTC791OgKAjh2ohlaPiznz'
    output = 'sources/downloadedBertModel/special_tokens_map.json'
    gdown.download(url, output, quiet=False)
    url = 'https://drive.google.com/uc?id=1-1KHBaHDaZzoh_T-aSZON7ZDj4huuJlm'
    output = 'sources/downloadedBertModel/model.safetensors'
    gdown.download(url, output, quiet=False)
    url = 'https://drive.google.com/uc?id=1-06dudeykLVAvCU5sjdE7eceJXGKWl_F'
    output = 'sources/downloadedBertModel/config.json'
    gdown.download(url, output, quiet=False)
current_dir = os.path.dirname(os.path.abspath(__file__))
huggingFaceModelPath = os.path.join(current_dir, "sources","downloadedBertModel")
if os.path.exists(huggingFaceModelPath):
    tokenizer = AutoTokenizer.from_pretrained(huggingFaceModelPath)
    bert = AutoModel.from_pretrained(huggingFaceModelPath)
else : 
    if not os.path.exists(huggingFaceModelPath):
        os.makedirs(huggingFaceModelPath)   
    downloadHuggingFaceBertModel()
    tokenizer = AutoTokenizer.from_pretrained(huggingFaceModelPath)
    bert = AutoModel.from_pretrained(huggingFaceModelPath)
    
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

# Load pre-trained weights
current_dir = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(current_dir, "sources","outputBert.pt")
if os.path.exists(path):
    model.load_state_dict(torch.load(path))
else : 
    url = 'https://drive.google.com/uc?id=1-0K3lDa4BdQPgJCCDhMRHwhI53xdR4Qx'
    output = 'sources/outputBert.pt'
    gdown.download(url, output, quiet=False)
    model.load_state_dict(torch.load(output))


# Load stop words
path = os.path.join(current_dir, "sources","stop.tr.turkish-lucene.txt")
if os.path.exists(path):
     with open(path, "r", encoding="utf-8") as file:
        stop_words_list = file.read().splitlines()
else : 
    url = 'https://drive.google.com/uc?id=1M3unCE4CqHYfFDxMAMx8nUZ8DJqwkOi3'
    output = 'sources/stop.tr.turkish-lucene.txt'
    gdown.download(url, output, quiet=False)
    with open(output, "r", encoding="utf-8") as file:
        stop_words_list = file.read().splitlines()
# Function to clean the text
def clean(text):
    punctuation_no_space = re.compile(r"(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})|(‘)|(“)|(”)|(°)|(\')")
    punctuation_with_space = re.compile(r"(<br\s/><br\s/?)|(-)|(/)|(:)")
    html_tags = r'<.*?>'
    line_cleaned = punctuation_no_space.sub("", text.lower())
    line_cleaned = punctuation_with_space.sub(" ", line_cleaned)
    line_cleaned = re.sub(html_tags, "", line_cleaned)
    line_cleaned = line_cleaned.split()
    line_cleaned = [word for word in line_cleaned if word not in stop_words_list]
    return ' '.join(line_cleaned)

@app.get("/test")
def testString(testText: str): 
    testText = testText.encode('utf-8').decode('utf-8')
    testMetin = clean(testText)
    MAX_LENGHT = 15
    tokens_unseen = tokenizer.batch_encode_plus(
        [testMetin],  
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