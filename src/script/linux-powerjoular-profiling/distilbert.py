'''
Author: kryst4lskyxx 906222327@qq.com
Date: 2023-10-17 16:49:03
LastEditors: kryst4lskyxx 906222327@qq.com
LastEditTime: 2023-10-18 19:50:35
FilePath: /Green Lab/experiment-runner/sample_code/bert.py
Description: This is the default configuration. Please set `customMade` and open koroFileHeader to view the configuration for settings: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import time
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import numpy as np

# Record the start time
start_time = time.time()

data = pd.read_csv("./SST-2/dev.tsv", sep='\t')

sentence = data['sentence'].to_numpy()

# Or load the DistilBERT model
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

sentiment_labels = []

for sen in sentence:

    # Or use DistilBERT for inference
    inputs = distilbert_tokenizer(sen, return_tensors="pt")
    output = distilbert_model(**inputs)
    logits = output.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    sentiment_labels.append(predicted_label)

# Record the end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Record the start time
start_time = time.time()

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from datasets import load_dataset

dataset = load_dataset("glue", "stsb")

# Load the DistilBERT model and tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Store similarity scores in lists
similarity_scores_distilbert = []

for example in dataset["train"]:
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]

    # Use DistilBERT for inference
    inputs = distilbert_tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)
    output = distilbert_model(**inputs)
    logits = output.logits
    similarity_score_distilbert = logits.softmax(dim=1)[0][1].item() * 5
    similarity_scores_distilbert.append(similarity_score_distilbert)

# Record the end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Record the start time
start_time = time.time()

from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

# Load the MNLI dataset
dataset = load_dataset("multi_nli", split="validation_matched")
desired_fraction = 0.3
split = dataset.train_test_split(test_size=desired_fraction)

# Get the "train" subset with the specified fraction

dataset = split["train"]

# Load the BERT model and tokenizer
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Load the DistilBERT model and tokenizer
distilbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Create a natural language inference pipeline
nli_pipeline_distilbert = pipeline("text-classification", model=distilbert_model, tokenizer=distilbert_tokenizer)

# Perform natural language inference on the MNLI dataset
results_distilbert = nli_pipeline_distilbert([example["premise"] for example in dataset], hypothesis=[example["hypothesis"] for example in dataset])

# Record the end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Exit the script
sys.exit()
