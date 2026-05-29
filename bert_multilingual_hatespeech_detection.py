import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

"""Install and Imports"""

!pip install transformers datasets scikit-learn seaborn wordcloud

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ NEW (for speed)
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

"""Load Dataset"""

from datasets import load_dataset

ds = load_dataset("tdavidson/hate_speech_offensive")

df1 = ds['train'].to_pandas()
df1.to_csv("hate_speech_offensive.csv", index=False)

df1 = pd.read_csv("hate_speech_offensive.csv")
df2 = pd.read_csv("Multilingual_dataset.csv")

df1.head()

df2.head()

df = pd.concat([df1, df2], ignore_index=True)

"""Standarization of df1 nad df2"""

df1 = df1[['tweet', 'class']]
df1 = df1.rename(columns={'tweet': 'text', 'class': 'label'})
df1['label'] = df1['label'].apply(lambda x: 0 if x == 2 else 1)

df2 = df2[['text', 'hate_label']]
df2 = df2.rename(columns={'hate_label': 'label'})

df = pd.concat([df1, df2], ignore_index=True)

df = df.dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ✅ CRITICAL FIX
df = df[df['label'].isin([0,1])]
df['label'] = df['label'].astype(int)

# smaller for speed
df = df.sample(4000, random_state=42)

print("Labels:", df['label'].unique())

"""TOKEN NORMALIZATION"""

def normalize_text(text):
    text = text.lower()

    # remove urls
    text = re.sub(r'http\S+|www\S+', '', text)

    # remove mentions
    text = re.sub(r'@\w+', '', text)

    # expand contractions
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"can't", "cannot", text)

    # reduce repeated characters
    text = re.sub(r'(.)\1+', r'\1\1', text)

    # remove special chars
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text

df['text'] = df['text'].apply(normalize_text)

"""EDA"""

# Class distribution
sns.countplot(x='label', data=df)
plt.title("Class Distribution")
plt.show()

# Text length
df['length'] = df['text'].apply(len)
sns.histplot(df['length'], bins=50)
plt.title("Text Length Distribution")
plt.show()

# Wordcloud
wc = WordCloud(width=800, height=400).generate(" ".join(df['text']))
plt.imshow(wc)
plt.axis('off')
plt.show()

"""Train test split"""

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

"""DATASET CLASS"""

class HateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

"""MODEL 1: BERT + HYPERTUNING

TOKENIZER + DATALOADER
"""

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = HateDataset(train_texts, train_labels, bert_tokenizer)
val_dataset = HateDataset(val_texts, val_labels, bert_tokenizer)

train_loader_bert = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader_bert = DataLoader(val_dataset, batch_size=32)

model_bert = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
).to(device)

"""training setup"""

optimizer = AdamW(model_bert.parameters(), lr=5e-5, weight_decay=0.01)

total_steps = len(train_loader_bert) * 4

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss()

# ✅ NEW
scaler = GradScaler()

def train_model(model, train_loader, val_loader):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()

    train_losses, val_losses = [], []

    for epoch in range(3):
        print(f"\nEpoch {epoch+1}")
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            inputs = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # ✅ FIXED autocast
            with autocast():
                outputs = model(inputs, attention_mask=mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        train_losses.append(total_loss/len(train_loader))

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with autocast():
                    outputs = model(inputs, attention_mask=mask, labels=labels)
                    loss = outputs.loss

                val_loss += loss.item()

        val_losses.append(val_loss/len(val_loader))

        print("Train Loss:", train_losses[-1])
        print("Val Loss:", val_losses[-1])

    return train_losses, val_losses

bert_train_loss, bert_val_loss = train_model(model_bert, train_loader_bert, val_loader_bert)

"""Model2: XLM-R

tokenizer and model
"""

xlm_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

train_dataset_xlm = HateDataset(train_texts, train_labels, xlm_tokenizer)
val_dataset_xlm = HateDataset(val_texts, val_labels, xlm_tokenizer)

train_loader_xlm = DataLoader(train_dataset_xlm, batch_size=32, shuffle=True)
val_loader_xlm = DataLoader(val_dataset_xlm, batch_size=32)

model_xlm = XLMRobertaForSequenceClassification.from_pretrained(
    'xlm-roberta-base',
    num_labels=2
).to(device)

xlm_train_loss, xlm_val_loss = train_model(model_xlm, train_loader_xlm, val_loader_xlm)

"""model 3:hybrid (BERT + LSTM)"""

from transformers import BertModel

class BertLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(768, 128, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = self.dropout(x)
        return self.fc(x)

model_hybrid = BertLSTM().to(device)
optimizer = AdamW(model_hybrid.parameters(), lr=2e-5)

for epoch in range(3):
    model_hybrid.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        inputs = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model_hybrid(inputs, mask)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

"""Evaluation function"""

def evaluate_model(model, val_loader):
    model.eval()

    preds, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs, attention_mask=mask) if hasattr(model, 'bert') == False else model(inputs, mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print("Accuracy:", acc)
    print(classification_report(true_labels, preds))

    return acc, true_labels, preds

val_loader = DataLoader(val_dataset, batch_size=32)

bert_acc, y_true_bert, y_pred_bert = evaluate_model(model_bert, val_loader)

val_loader = DataLoader(val_dataset_xlm, batch_size=32)

xlm_acc, y_true_xlm, y_pred_xlm = evaluate_model(model_xlm, val_loader)

hybrid_acc, y_true_hybrid, y_pred_hybrid = evaluate_model(model_hybrid, val_loader)

"""loss graph"""

plt.plot(bert_train_loss, label='BERT Train')
plt.plot(bert_val_loss, label='BERT Val')

plt.plot(xlm_train_loss, label='XLM Train')
plt.plot(xlm_val_loss, label='XLM Val')

plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

"""confusion matrix"""

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.show()

plot_confusion(y_true_bert, y_pred_bert, "BERT")
plot_confusion(y_true_xlm, y_pred_xlm, "XLM-R")
plot_confusion(y_true_hybrid, y_pred_hybrid, "Hybrid")

"""model comparision graph"""

models = ['BERT', 'XLM-R', 'Hybrid']
accuracies = [bert_acc, xlm_acc, hybrid_acc]

sns.barplot(x=models, y=accuracies)
plt.title("Accuracy Comparison")
plt.show()

"""F1- score commparison"""

f1_scores = [
    f1_score(y_true_bert, y_pred_bert, average='weighted'),
    f1_score(y_true_xlm, y_pred_xlm, average='weighted'),
    f1_score(y_true_hybrid, y_pred_hybrid, average='weighted')
]

sns.barplot(x=models, y=f1_scores)
plt.title("F1 Score Comparison")
plt.show()
