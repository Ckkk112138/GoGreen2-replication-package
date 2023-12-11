'''
Author: kryst4lskyxx 906222327@qq.com
Date: 2023-10-17 16:49:03
LastEditors: kryst4lskyxx 906222327@qq.com
LastEditTime: 2023-10-18 19:50:35
FilePath: /Green Lab/experiment-runner/sample_code/bert.py
Description: This is the default setting, please set `customMade`, open koroFileHeader to view the configuration for settings: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
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

# Load the BERT model
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

sentiment_labels = []

for sen in sentence:
    # Use BERT for inference
    inputs = bert_tokenizer(sen, return_tensors="pt")
    output = bert_model(**inputs)
    logits = output.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    sentiment_labels.append(predicted_label)

# Record the end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Record the start time
start_time = time.time()

from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline
from datasets import load_dataset

dataset = load_dataset("glue", "stsb")

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Store similarity scores in lists
similarity_scores_bert = []
similarity_scores_distilbert = []

for example in dataset["train"]:
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]

    # Use BERT for inference
    inputs = bert_tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)
    output = bert_model(**inputs)
    logits = output.logits
    similarity_score_bert = logits.softmax(dim=1)[0][1].item() * 5
    similarity_scores_bert.append(similarity_score_bert)

# Record the end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print(f"STS-B code execution time: {execution_time} seconds")

from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from datasets import load_dataset

# Record the start time
start_time = time.time()

# Load MNLI dataset
dataset = load_dataset("multi_nli", split="validation_matched")
desired_fraction = 0.3
split = dataset.train_test_split(test_size=desired_fraction)

# Get the "train" subset with the specified fraction

dataset = split["train"]

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Create a natural language inference pipeline
nli_pipeline_bert = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)

# Perform natural language inference on the MNLI dataset
results_bert = nli_pipeline_bert([example["premise"] for example in dataset], hypothesis=[example["hypothesis"] for example in dataset])

# Record the end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Terminate the script
sys.exit()
