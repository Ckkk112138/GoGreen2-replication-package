'''
Author: kryst4lskyxx 906222327@qq.com
Date: 2023-10-17 16:49:03
LastEditors: kryst4lskyxx 906222327@qq.com
LastEditTime: 2023-10-18 19:50:35
FilePath: /Green Lab/experiment-runner/sample_code/bert.py
Description: default setting `customMade`, using koroFileHeader to check configuration (https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE)
'''
import torch
import time
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import numpy as np

''' SST-2 Dataset '''
data = pd.read_csv("./SST-2/train.tsv", sep='\t')

sentence = data['sentence'].to_numpy()

# Load BERT Model
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Load DistilBERT Model
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased")
distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased")

# Initilize sentiment labels
sentiment_labels = []

for sen in sentence:
    # Generate text through BERT
    inputs = bert_tokenizer(sen, return_tensors="pt")
    output = bert_model(**inputs)
    logits = output.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    sentiment_labels.append(predicted_label)

    # Generate through DistilBERT
    inputs = distilbert_tokenizer(sentence, return_tensors="pt")
    output = distilbert_model(**inputs)
    logits = output.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    sentiment_labels.append(predicted_label)

# Output result
for i, label in enumerate(sentiment_labels):
    print(f"Sentence {i + 1}: Sentiment Label {label}")

time.sleep(60)

''' STS-B Dataset '''
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline
from datasets import load_dataset

dataset = load_dataset("stsb_multi_mt", name="en", split="train")

# Load BERT Model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Load DistilBERT Model and tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Initialize similarity scores
similarity_scores_bert = []
similarity_scores_distilbert = []

for example in dataset:
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]

    # Inference through BERT
    inputs = bert_tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)
    output = bert_model(**inputs)
    logits = output.logits
    similarity_score_bert = logits.softmax(dim=1)[0][1].item() * 5
    similarity_scores_bert.append(similarity_score_bert)

    # Inference through DistilBERT
    inputs = distilbert_tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)
    output = distilbert_model(**inputs)
    logits = output.logits
    similarity_score_distilbert = logits.softmax(dim=1)[0][1].item() * 5
    similarity_scores_distilbert.append(similarity_score_distilbert)

# Export similarity scores
for i, (score_bert, score_distilbert) in enumerate(zip(similarity_scores_bert, similarity_scores_distilbert)):
    print(f"Example {i + 1} - Similarity Score (BERT): {score_bert:.2f}, Similarity Score (DistilBERT): {score_distilbert:.2f}")


''' MNLI Dataset '''
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from datasets import load_dataset

# Load MNLI dataset
dataset = load_dataset("multi_nli", split="validation_matched")

# Load BERT Model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Load DistilBERT Model and tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Create pipeline for text classification
nli_pipeline_bert = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)
nli_pipeline_distilbert = pipeline("text-classification", model=distilbert_model, tokenizer=distilbert_tokenizer)

# Process text classification
results_bert = nli_pipeline_bert([example["premise"] for example in dataset], hypothesis=[example["hypothesis"] for example in dataset])
results_distilbert = nli_pipeline_distilbert([example["premise"] for example in dataset], hypothesis=[example["hypothesis"] for example in dataset])

# Export result
for i, (result_bert, result_distilbert) in enumerate(zip(results_bert, results_distilbert)):
    premise = dataset[i]["premise"]
    hypothesis = dataset[i]["hypothesis"]
    label = dataset[i]["label"]
    prediction_bert = result_bert["label"]
    prediction_distilbert = result_distilbert["label"]
    
    print(f"Example {i + 1}:\nPremise: {premise}\nHypothesis: {hypothesis}\nActual Label: {label}\nBERT Prediction: {prediction_bert}\nDistilBERT Prediction: {prediction_distilbert}\n")
