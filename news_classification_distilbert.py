import sqlite3
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import pandas as pd
import os
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from torch.optim import AdamW

path = kagglehub.dataset_download("rmisra/news-category-dataset")
dataset_path = "/kaggle/input/news-category-dataset"
df = pd.read_json(f"{dataset_path}/News_Category_Dataset_v3.json", lines=True)

selected_categories = ['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY']
selected_rows = df.loc[df['category'].isin(selected_categories)]
new_df = selected_rows[['headline', 'category']].copy()
new_df.reset_index(drop=True, inplace=True)  # إعادة ترتيب الفهرس
y = new_df["category"]
y.value_counts().plot.pie(autopct='%.2f')

undersample = RandomUnderSampler(random_state=42)
X = new_df["headline"].to_numpy().reshape(-1, 1)
Y = new_df["category"]
X_under, Y_under = undersample.fit_resample(X, Y)
df_under = pd.DataFrame({
    "headline": X_under.flatten(),
    "category": Y_under
})
y = df_under["category"]
y.value_counts().plot.pie(autopct='%.2f')

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stop_words = ENGLISH_STOP_WORDS

def clean(lines):
    punctuation_no_space = re.compile(r"(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})|(‘)|(“)|(”)|(°)|(\')")
    punctuation_with_space = re.compile(r"(<br\s*/?><br\s*/?>)|(-)|(/)|(:)")
    html_tags = r'<.*?>'

    cleaned_lines = []
    for line in lines:
        s = line.lower()
        s = punctuation_no_space.sub("", s)
        s = punctuation_with_space.sub(" ", s)
        s = re.sub(html_tags, "", s)
        tokens = s.split()
        tokens = [w for w in tokens if w not in stop_words]

        cleaned_lines.append(" ".join(tokens))

    return cleaned_lines

df_under['Cleaned_headline'] = clean(df_under['headline'])
df_under.head()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

df_under['label_num'] = df_under['category'].map({
    'POLITICS': 0,
    'WELLNESS': 1,
    'ENTERTAINMENT': 2,
    'TRAVEL': 3,
    'STYLE & BEAUTY': 4
})

x = df_under['headline']
y = df_under['label_num']
train_text, temp_text, train_labels, temp_labels = train_test_split(x, y,
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=y)
# Validation-Test split
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

MAX_LENGTH = 13

tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

train_y = torch.tensor(train_labels.tolist())
val_y = torch.tensor(val_labels.tolist())
test_y = torch.tensor(test_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
batch_size = 64
train_data = TensorDataset(tokens_train['input_ids'], tokens_train['attention_mask'], train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(tokens_val['input_ids'], tokens_val['attention_mask'], val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

for param in bert.parameters():
    param.requires_grad = False

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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        sent_emb = self.mean_pool(last_hidden, attention_mask)
        x = self.fc1(sent_emb)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.logsoftmax(x)
        return x


model = DistilBERT_Arch(bert, num_labels=5)

optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

cross_entropy = nn.NLLLoss()
epochs = 15

def train():
  model.train()
  total_loss, total_accuracy = 0, 0

  for step,batch in enumerate(train_dataloader):
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
    batch = [r for r in batch]
    sent_id, mask, labels = batch
    model.zero_grad()
    preds = model(sent_id, mask)
    loss = cross_entropy(preds, labels)
    total_loss = total_loss + loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    preds=preds.detach().cpu().numpy()

  avg_loss = total_loss / len(train_dataloader)

  return avg_loss

def evaluate():
  print("\nEvaluating...")
  model.eval()
  total_loss, total_accuracy = 0, 0
  for step,batch in enumerate(val_dataloader):
    if step % 50 == 0 and not step == 0:


      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    batch = [t for t in batch]
    sent_id, mask, labels = batch
    with torch.no_grad():
      preds = model(sent_id, mask)
      loss = cross_entropy(preds,labels)
      total_loss = total_loss + loss.item()
      preds = preds.detach().cpu().numpy()
  avg_loss = total_loss / len(val_dataloader)
  return avg_loss

weights_path = '/content/drive/MyDrive/NewsClassification/outputBert.pt'

# Train and predict
best_valid_loss = float('inf')
train_losses = []
valid_losses = []

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss = train()
    valid_loss = evaluate()
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), weights_path)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
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
path = '/content/drive/MyDrive/NewsClassification/outputBert.pt'
model.load_state_dict(torch.load(path))
with torch.no_grad():
  preds = model(tokens_test['input_ids'], tokens_test['attention_mask'])
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
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
path = '/content/drive/MyDrive/NewsClassification/outputBert.pt'
model.load_state_dict(torch.load(path))


stop_words = ENGLISH_STOP_WORDS
def clean(lines):
    punctuation_no_space = re.compile(r"(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})|(‘)|(“)|(”)|(°)|(\')")
    punctuation_with_space = re.compile(r"(<br\s*/?><br\s*/?>)|(-)|(/)|(:)")
    html_tags = r'<.*?>'

    cleaned_lines = []
    for line in lines:
        s = line.lower()
        s = punctuation_no_space.sub("", s)
        s = punctuation_with_space.sub(" ", s)
        s = re.sub(html_tags, "", s)
        tokens = s.split()
        tokens = [w for w in tokens if w not in stop_words]

        cleaned_lines.append(" ".join(tokens))

    return cleaned_lines

testText = ["US politics in brief: Bowser keeps calm, Democrats rage", #POLITICS (0)
                    "Helping women navigate perimenopause",#WELLNESS (1)
            "'Mamma Mia!' returns to Broadway after a decade away, bringing the dance party back to New York",#ENTERTAINMENT (2)
            "Air India Suspends Washington Service From Sep 2025",#TRAVEL (3)
            "Sydney Sweeney Looks So Different With New Curly Hair and Bangs"#STYLE & BEAUTY (4)
                    ]
testText = clean(testText)

MAX_LENGHT = 13
tokens_unseen = tokenizer.batch_encode_plus(
    testText,
    max_length = MAX_LENGHT,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

with torch.no_grad():
  preds = model(tokens_unseen['input_ids'],tokens_unseen['attention_mask'])
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
preds
