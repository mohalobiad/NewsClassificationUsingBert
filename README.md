# News Classification Using BERT/DistilBERT (FastAPI + Docker)

A production-ready REST API that classifies news **headlines** into 5 categories using  DistilBERT with a custom classifier head.

## Dataset
- **Source:** News Category Dataset (v3) from Kaggle (rmisra).
- **Size/Schema:** ~209,527 rows with columns: `link, headline, category, short_description, authors, date`.
- **Task focus:** headline → category.
- **Classes used in this project (balanced):** `POLITICS`, `WELLNESS`, `ENTERTAINMENT`, `TRAVEL`, `STYLE & BEAUTY`.
- **Class imbalance handling:** undersampling to obtain a balanced set.

## Model
- **Backbone:** DistilBERT (`distilbert-base-uncased`), BERT weights **frozen**.
- **Head:** Mean pooling over last hidden states → `Linear( hidden → 512 )` + ReLU + Dropout → `Linear( 512 → 5 )` + LogSoftmax.
- **Tokenizer:** DistilBERT tokenizer.

## Training (reference configuration)
- **Batch size:** 32  
- **Optimizer:** AdamW (weight_decay=0.01)  
- **LR:** 1e-5  
- **Epochs:** 8  
- **Loss:** NLLLoss over LogSoftmax outputs  
- **Weights file:** `sources/outputBert.pt` (PyTorch state_dict)

> Note: To load weights, you must reconstruct the exact model class before `load_state_dict`.

Project Structure
├── Dockerfile
├── requirements.txt
├── index.html
├── News.py                       # FastAPI app
├── news_classification_distilbert.py   # Training / eval code
├── sources/
├── outputBert.pt                  # Trained weights
├── english_stopwords.txt
├── downloadedBertModel/
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── model.safetensors
│   └── config.json

## bash

```bash
docker pull swr.tr-west-1.myhuaweicloud.com/cloud-workspace/news-app:latest
docker run -p 8000:8000 swr.tr-west-1.myhuaweicloud.com/cloud-workspace/news-app:latest
```
** Once the container is running, open:
http://localhost:8000
You’ll see a simple web page with:
A text box to enter a news headline.
A button to send the text and get the classification result.
Several sample buttons to quickly test the model.

## Reproduce Training
- Download dataset (Kaggle) and prepare subset of classes:
- Keep: POLITICS, WELLNESS, ENTERTAINMENT, TRAVEL, STYLE & BEAUTY.
- Undersample to balance classes.
- Tokenize headlines (max_length=13), create DataLoaders (train/val/test, batch_size=32).
- Build model with frozen DistilBERT + custom head; train with AdamW (lr=1e-5, epochs=8).
- Save weights to sources/outputBert.pt.
- Evaluate with accuracy, precision/recall/F1 and confusion matrix.

## Results (reference)
- Accuracy around ~81% on the held-out test set for the 5-class configuration.
  
## Notes & Tips :
- Always rebuild the model class before loading .pt state dicts.

## For more information, read my article that explains everything about this project on my Medium account:
https://medium.com/@moh.alobiad
